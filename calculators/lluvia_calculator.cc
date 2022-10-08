#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#include <lluvia/core.h>

#include "mediapipe/lluvia-mediapipe/calculators/lluvia_calculator.pb.h"

#include <memory>
#include <tuple>

namespace mediapipe {


struct PortHandler {
    std::shared_ptr<ll::Buffer> stagingBuffer;
    std::unique_ptr<uint8_t [], ll::Buffer::BufferMapDeleter> stagingBufferMappedPtr;
    std::shared_ptr<ll::Image> image;
    std::shared_ptr<ll::ImageView> imageView;
};

// Calculator to pass a CPU image through. It prints in the logs the
// image attributes such as resolution and format.
class LluviaCalculator : public CalculatorBase {
public:
    static ::mediapipe::Status GetContract(CalculatorContract* cc);
    ::mediapipe::Status Open(CalculatorContext* cc) override;
    ::mediapipe::Status Process(CalculatorContext* cc) override;
    ::mediapipe::Status Close(CalculatorContext* cc) override;

    

private:
    ::mediapipe::Status InitNode(const ImageFrame*);
    ::mediapipe::Status InitInputImage(const ImageFrame*);

    std::tuple<bool, ll::ChannelCount, ll::ChannelType> getLluviaImageFormat(const mediapipe::ImageFormat_Format format);
    std::tuple<bool, mediapipe::ImageFormat_Format> getMediapipeImageFormat(const ll::ChannelCount channelCount, const ll::ChannelType channelType);

    lluvia::LluviaCalculatorOptions m_options;

    std::shared_ptr<ll::Session> m_session {};
    std::shared_ptr<ll::Memory> m_hostMemory {};
    std::shared_ptr<ll::Memory> m_deviceMemory {};

    std::unique_ptr<ll::CommandBuffer> m_cmdBuffer {};

    // the container node
    std::shared_ptr<ll::ContainerNode> m_containerNode {};

    // TODO: support more than one input and output
    mediapipe::PortHandler m_inputHandler;
    mediapipe::PortHandler m_outputHandler;

    std::once_flag m_configureNode {};
};

::mediapipe::Status LluviaCalculator::GetContract(CalculatorContract* cc) {

    LOG(INFO) << "LLUVIA: GetContract()";

    // FIXME: try to support GPUBuffer instead. Use AnyOf
    // cc->Inputs().Index(0).Set<ImageFrame>();
    cc->Inputs().Tag("IN_0").Set<ImageFrame>();

    cc->Outputs().Tag("OUT_0").Set<ImageFrame>();
    return ::mediapipe::OkStatus();
}

::mediapipe::Status LluviaCalculator::Open(CalculatorContext* cc) {

    LOG(INFO) << "LLUVIA: Open()";

    m_options = cc->Options<lluvia::LluviaCalculatorOptions>();

    // try to find a DISCRETE_GPU device
    auto availableDevices = ll::Session::getAvailableDevices();
    auto selectedDevice = availableDevices[0];

    for (const auto& desc : availableDevices) {
        if (desc.deviceType == ll::DeviceType::DiscreteGPU) {
            selectedDevice = desc;
        }
    }

    LOG(INFO) << "LLUVIA: using device: " << selectedDevice.name;

    // TODO: options for creating the session
    // - device type
    auto sessionDescriptor = ll::SessionDescriptor()
        .setDeviceDescriptor(selectedDevice)
        .enableDebug(m_options.enable_debug());

    m_session = ll::Session::create(sessionDescriptor);
    
    m_hostMemory = m_session->getHostMemory();

    auto deviceMemoryProperties = ll::MemoryPropertyFlagBits::DeviceLocal;
    m_deviceMemory = m_session->createMemory(deviceMemoryProperties, 32 * 1024 * 1024, false);

    // LOG(INFO) << "LLUVIA: host memory created: " << m_hostMemory->getPageSize();
    // LOG(INFO) << "LLUVIA: device memory created: " << m_hostMemory->getPageSize();

    // load all supplied libraries to the session
    for (auto i = 0; i < m_options.library_path_size(); ++i) {
        // LOG(INFO) << "LLUVIA: library path: " << m_options.library_path(i);
        m_session->loadLibrary(m_options.library_path(i));
    }

    // execute all the scripts in the session
    for (auto i = 0; i < m_options.script_path_size(); ++i) {
        // LOG(INFO) << "LLUVIA: script path: " << m_options.script_path(i);
        m_session->scriptFile(m_options.script_path(i));
    }

    for (const auto& desc : m_session->getNodeBuilderDescriptors()) {
        LOG(INFO) << "LLUVIA: " << desc.name;
    }

    return ::mediapipe::OkStatus();
}

::mediapipe::Status LluviaCalculator::InitNode(const ImageFrame* inputImage) {

    LOG(INFO) << "LLUVIA: InitNode() start";

    const auto width = inputImage->Width();
    const auto height = inputImage->Height();

    const auto initImageStatus = InitInputImage(inputImage);
    if (initImageStatus != ::mediapipe::OkStatus()) {
        return initImageStatus;
    }

    // LOG(INFO) << "LLUVIA: InitNode() width: " << width << " height: " << height;

    m_containerNode = m_session->createContainerNode(m_options.container_node());

    ///////////////////////////////////////////////////////////////////////////
    // Port bindings
    // FIXME: Use input_port_bindings
    m_containerNode->bind("in_image", m_inputHandler.imageView);

    ///////////////////////////////////////////////////////////////////////////
    // Parameters
    // TODO

    ///////////////////////////////////////////////////////////////////////////
    // Node init
    m_containerNode->init();

    ///////////////////////////////////////////////////////////////////////////
    // Outputs
    m_outputHandler.imageView = std::static_pointer_cast<ll::ImageView>(m_containerNode->getPort("out_image"));
    m_outputHandler.image = m_outputHandler.imageView->getImage();
    m_outputHandler.stagingBuffer = m_hostMemory->createBuffer(m_outputHandler.image->getMinimumSize());
    m_outputHandler.stagingBufferMappedPtr = m_outputHandler.stagingBuffer->map<uint8_t []>();

    ///////////////////////////////////////////////////////////////////////////
    // Command buffer
    m_cmdBuffer = m_session->createCommandBuffer();
    m_cmdBuffer->begin();

    // Copy staging buffer to m_inputImage. Consider changing image layout.
    m_cmdBuffer->changeImageLayout(*m_inputHandler.image, ll::ImageLayout::TransferDstOptimal);
    m_cmdBuffer->memoryBarrier();
    m_cmdBuffer->copyBufferToImage(*m_inputHandler.stagingBuffer, *m_inputHandler.image);
    m_cmdBuffer->memoryBarrier(); // TODO: needed?
    m_cmdBuffer->changeImageLayout(*m_inputHandler.image, ll::ImageLayout::General);
    m_cmdBuffer->memoryBarrier();

    // Compute
    m_cmdBuffer->run(*m_containerNode);
    m_cmdBuffer->memoryBarrier();

    // Copy output image to staging buffer
    m_cmdBuffer->changeImageLayout(*m_outputHandler.image, ll::ImageLayout::TransferSrcOptimal);
    m_cmdBuffer->memoryBarrier();
    m_cmdBuffer->copyImageToBuffer(*m_outputHandler.image, *m_outputHandler.stagingBuffer);
    m_cmdBuffer->memoryBarrier();
    m_cmdBuffer->changeImageLayout(*m_outputHandler.image, ll::ImageLayout::General);
    m_cmdBuffer->memoryBarrier();

    m_cmdBuffer->end();

    LOG(INFO) << "LLUVIA: InitNode() finish";

    return ::mediapipe::OkStatus();
}

::mediapipe::Status LluviaCalculator::InitInputImage(const ImageFrame* inputImage) {

    const auto width = inputImage->Width();
    const auto height = inputImage->Height();

    // TODO: usage flags
    m_inputHandler.stagingBuffer = m_hostMemory->createBuffer(static_cast<uint64_t>(inputImage->PixelDataSize()));
    m_inputHandler.stagingBufferMappedPtr = m_inputHandler.stagingBuffer->map<uint8_t []>();

    const ll::ImageUsageFlags imgUsageFlags = { ll::ImageUsageFlagBits::Storage
                                                | ll::ImageUsageFlagBits::Sampled
                                                | ll::ImageUsageFlagBits::TransferDst
                                                | ll::ImageUsageFlagBits::TransferSrc};

    auto imageFormatSupported = false;
    auto channelCount = ll::ChannelCount::C1;
    auto channelType = ll::ChannelType::Uint8;

    std::tie(imageFormatSupported, channelCount, channelType) = getLluviaImageFormat(inputImage->Format());

    if (!imageFormatSupported) {
        return ::mediapipe::UnknownError("image format not supported");
    }

    const auto imgDesc = ll::ImageDescriptor{1, static_cast<uint32_t >(height), static_cast<uint32_t>(width),
                                                channelCount, channelType}
                .setUsageFlags(imgUsageFlags);

    m_inputHandler.image = m_deviceMemory->createImage(imgDesc);
    m_inputHandler.imageView = m_inputHandler.image->createImageView(ll::ImageViewDescriptor{ll::ImageAddressMode::ClampToBorder,
                                                                                ll::ImageFilterMode::Nearest,
                                                                                false,
                                                                                false});


    m_inputHandler.image->changeImageLayout(ll::ImageLayout::General);

    return ::mediapipe::OkStatus();
}

::mediapipe::Status LluviaCalculator::Process(CalculatorContext* cc) {


    auto& inputImage = cc->Inputs().Tag("IN_0").Get<ImageFrame>();

    // init the internals given a concrete ImageFrame
    std::call_once(m_configureNode, &LluviaCalculator::InitNode, this, &inputImage);

    // Copy input image to a staging buffer
    inputImage.CopyToBuffer(&m_inputHandler.stagingBufferMappedPtr[0], m_inputHandler.stagingBuffer->getSize());

    m_session->run(*m_cmdBuffer);

    ::mediapipe::ImageFormat_Format outputImageFormat = ::mediapipe::ImageFormat_Format_UNKNOWN;
    bool imageFormatFound = false;

    std::tie(imageFormatFound, outputImageFormat) =  getMediapipeImageFormat(m_outputHandler.image->getChannelCount(), m_outputHandler.image->getChannelType());
    if (!imageFormatFound) {
        return ::mediapipe::UnknownError("unable to find compatible output image format");
    }

    std::unique_ptr<ImageFrame> outputImage = absl::make_unique<ImageFrame>(outputImageFormat, m_outputHandler.image->getWidth(), m_outputHandler.image->getHeight());

    // copy staging buffer to output ImageFrame
    std::memcpy(outputImage->MutablePixelData(), &m_outputHandler.stagingBufferMappedPtr[0], m_outputHandler.stagingBuffer->getSize());

    LOG_EVERY_N(INFO, 300) << "LluviaCalculator: shape [h:"
                            << std::to_string(outputImage->Height()) << ", w:" << std::to_string(outputImage->Width()) << "], format: "
                            << std::to_string(static_cast<int>(outputImage->Format())) << ", channel size: "
                            << std::to_string(outputImage->ChannelSize());

    cc->Outputs().Tag("OUT_0").Add(outputImage.release(), cc->InputTimestamp());


    return ::mediapipe::OkStatus();
}

::mediapipe::Status LluviaCalculator::Close(CalculatorContext* cc) {
    return ::mediapipe::OkStatus();
}

std::tuple<bool, ll::ChannelCount, ll::ChannelType> LluviaCalculator::getLluviaImageFormat(const mediapipe::ImageFormat_Format format) {

    switch(format) {

        case ImageFormat_Format_SRGB:
            return std::make_tuple(true, ll::ChannelCount::C3, ll::ChannelType::Uint8);
        
        case ImageFormat_Format_SRGBA:
            return std::make_tuple(true, ll::ChannelCount::C4, ll::ChannelType::Uint8);
        
        case ImageFormat_Format_SBGRA:
            return std::make_tuple(true, ll::ChannelCount::C4, ll::ChannelType::Uint8);
        
        case ImageFormat_Format_GRAY8:
            return std::make_tuple(true, ll::ChannelCount::C1, ll::ChannelType::Uint8);
        
        case ImageFormat_Format_GRAY16:
            return std::make_tuple(true, ll::ChannelCount::C1, ll::ChannelType::Uint16);
        
        case ImageFormat_Format_SRGBA64:
            return std::make_tuple(true, ll::ChannelCount::C4, ll::ChannelType::Uint64);
        
        case ImageFormat_Format_VEC32F1:
            return std::make_tuple(true, ll::ChannelCount::C1, ll::ChannelType::Float32);
        
        case ImageFormat_Format_VEC32F2:
            return std::make_tuple(true, ll::ChannelCount::C2, ll::ChannelType::Float32);
        
        default:
            return std::make_tuple(false, ll::ChannelCount::C1, ll::ChannelType::Uint8);
    }
}

std::tuple<bool, mediapipe::ImageFormat_Format> LluviaCalculator::getMediapipeImageFormat(const ll::ChannelCount channelCount, const ll::ChannelType channelType) {

    switch(channelCount) {

        case ll::ChannelCount::C1:
            switch (channelType) {
            case ll::ChannelType::Uint8:
                return std::make_tuple(true, ::mediapipe::ImageFormat_Format_GRAY8);
            case ll::ChannelType::Uint16:
                return std::make_tuple(true, ::mediapipe::ImageFormat_Format_GRAY16);
            case ll::ChannelType::Float32:
                return std::make_tuple(true, ::mediapipe::ImageFormat_Format_VEC32F1);
            default:
                return std::make_tuple(false, ::mediapipe::ImageFormat_Format_UNKNOWN);
            }

        case ll::ChannelCount::C2:
            switch (channelType) {
            case ll::ChannelType::Float32:
                return std::make_tuple(true, ::mediapipe::ImageFormat_Format_VEC32F2);
            default:
                return std::make_tuple(false, ::mediapipe::ImageFormat_Format_UNKNOWN);
            }

        case ll::ChannelCount::C3:
            switch (channelType) {
            case ll::ChannelType::Uint8:
                return std::make_tuple(true, ::mediapipe::ImageFormat_Format_SRGB);
            default:
                return std::make_tuple(false, ::mediapipe::ImageFormat_Format_UNKNOWN);
            }

        case ll::ChannelCount::C4:
            switch (channelType) {
            case ll::ChannelType::Uint8:
                return std::make_tuple(true, ::mediapipe::ImageFormat_Format_SRGBA);
            case ll::ChannelType::Uint64:
                return std::make_tuple(true, ::mediapipe::ImageFormat_Format_SRGBA64);
            default:
                return std::make_tuple(false, ::mediapipe::ImageFormat_Format_UNKNOWN);
            }
    }
}

REGISTER_CALCULATOR(LluviaCalculator);

}  // namespace mediapipe
