#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#define HAVE_GPU_BUFFER
#ifdef __APPLE__
#include "mediapipe/objc/util.h"
#endif

#include "mediapipe/gpu/gl_calculator_helper.h"

#include <lluvia/core.h>

#include "mediapipe/lluvia-mediapipe/calculators/lluvia_calculator.pb.h"

#include <memory>
#include <tuple>

namespace mediapipe {


struct PortHandler {
    // the staging buffer used to transfer data from ImageFrame or GPUBuffer to Lluvia/Vulkan memory space
    std::shared_ptr<ll::Buffer> stagingBuffer;
    
    // the pointer to the staging buffer mapped to the host memory space.
    std::unique_ptr<uint8_t [], ll::Buffer::BufferMapDeleter> stagingBufferMappedPtr;

    // the image in device memory
    std::shared_ptr<ll::Image> image;

    // the image vide to feed to the container node
    std::shared_ptr<ll::ImageView> imageView;

    // port name in the Lluvia's container node
    std::string lluviaPortName;

    // mediapipe tag used to bind port
    std::string mediapipeTag;

    // type of mediapipe packet expected to be received in this port.
    lluvia::MediapipePacketType mediapipePacketType;
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
    ::mediapipe::Status InitNode(CalculatorContext* cc);
    ::mediapipe::Status InitImageFramePort(const ImageFrame*, PortHandler& portHandler);

    std::tuple<bool, ll::ChannelCount, ll::ChannelType> getLluviaImageFormat(const mediapipe::ImageFormat_Format format);
    std::tuple<bool, mediapipe::ImageFormat_Format> getMediapipeImageFormat(const ll::ChannelCount channelCount, const ll::ChannelType channelType);

    lluvia::LluviaCalculatorOptions m_options;

    std::shared_ptr<ll::Session> m_session {};
    std::shared_ptr<ll::Memory> m_hostMemory {};
    std::shared_ptr<ll::Memory> m_deviceMemory {};

    std::unique_ptr<ll::CommandBuffer> m_cmdBuffer {};

    // the container node
    std::shared_ptr<ll::ContainerNode> m_containerNode {};

    std::vector<mediapipe::PortHandler> m_inputHandlers;
    std::vector<mediapipe::PortHandler> m_outputHandlers;

    std::once_flag m_configureNode {};
};

::mediapipe::Status LluviaCalculator::GetContract(CalculatorContract* cc) {

    LOG(INFO) << "LLUVIA: GetContract()";

    for (const auto& tag : cc->Inputs().GetTags()) {
        cc->Inputs().Tag(tag).SetOneOf<ImageFrame, GpuBuffer>();
    }

    for (const auto& tag : cc->Outputs().GetTags()) {
        cc->Outputs().Tag(tag).SetOneOf<ImageFrame, GpuBuffer>();
    }

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

::mediapipe::Status LluviaCalculator::InitNode(CalculatorContext* cc) {

    LOG(INFO) << "LLUVIA: InitNode() start";

    ///////////////////////////////////////////////////////////////////////////
    // Container node
    m_containerNode = m_session->createContainerNode(m_options.container_node());

    ///////////////////////////////////////////////////////////////////////////
    // Input bindings
    for (auto i = 0; i < m_options.input_port_binding_size(); ++i) {
        
        const auto& portBinding = m_options.input_port_binding(i);

        // initialize the port handler for with the protobuffer attributes
        auto portHandler = PortHandler {};
        portHandler.mediapipePacketType = portBinding.packet_type();
        portHandler.mediapipeTag = portBinding.mediapipe_tag();
        portHandler.lluviaPortName = portBinding.lluvia_port();

        // configure this port as receiving an ImageFrame
        if (portBinding.packet_type() == lluvia::IMAGE_FRAME) {

            auto& inputImage = cc->Inputs().Tag(portBinding.mediapipe_tag()).Get<ImageFrame>();
            
            const auto initImageStatus = InitImageFramePort(&inputImage, portHandler);
            if (initImageStatus != ::mediapipe::OkStatus()) {
                return initImageStatus;
            }
            
            // bind to the container node
            m_containerNode->bind(portHandler.lluviaPortName, portHandler.imageView);

            // finally, add the handler to the list of input handlers
            m_inputHandlers.push_back(std::move(portHandler));
        }

        // TODO: support lluvia::GPU_BUFFER

        
    }

    ///////////////////////////////////////////////////////////////////////////
    // Parameters
    // TODO

    ///////////////////////////////////////////////////////////////////////////
    // Node init
    m_containerNode->init();

    ///////////////////////////////////////////////////////////////////////////
    // Outputs
    for (auto i = 0; i < m_options.output_port_binding_size(); ++i) {

        const auto& portBinding = m_options.output_port_binding(i);

        // initialize the port handler for with the protobuffer attributes
        auto portHandler = PortHandler {};
        portHandler.mediapipePacketType = portBinding.packet_type();
        portHandler.mediapipeTag = portBinding.mediapipe_tag();
        portHandler.lluviaPortName = portBinding.lluvia_port();

        // initialize lluvia objects
        try {
            // getting unexisting port name throws exception
            portHandler.imageView = std::static_pointer_cast<ll::ImageView>(m_containerNode->getPort(portHandler.lluviaPortName));
        } catch(std::system_error& e) {
            return absl::UnknownError(e.what());
        }
        
        portHandler.image = portHandler.imageView->getImage();
        portHandler.stagingBuffer = m_hostMemory->createBuffer(portHandler.image->getMinimumSize());
        portHandler.stagingBufferMappedPtr = portHandler.stagingBuffer->map<uint8_t []>();

        // finally, add the handler to the list of output handlers
        m_outputHandlers.push_back(std::move(portHandler));
    }

    ///////////////////////////////////////////////////////////////////////////
    // Command buffer
    m_cmdBuffer = m_session->createCommandBuffer();
    m_cmdBuffer->begin();

    // Copy all staging buffers to their corresponding port handler image.
    for (auto& inputHandler : m_inputHandlers) {
        m_cmdBuffer->changeImageLayout(*inputHandler.image, ll::ImageLayout::TransferDstOptimal);
        m_cmdBuffer->memoryBarrier();
        m_cmdBuffer->copyBufferToImage(*inputHandler.stagingBuffer, *inputHandler.image);
        m_cmdBuffer->memoryBarrier(); // TODO: needed?
        m_cmdBuffer->changeImageLayout(*inputHandler.image, ll::ImageLayout::General);
        m_cmdBuffer->memoryBarrier();
    }
    
    // Compute
    m_cmdBuffer->run(*m_containerNode);
    m_cmdBuffer->memoryBarrier();

    // Copy all output images to their corresponding staging buffers
    for (auto& outputHandler : m_outputHandlers) {
        
        m_cmdBuffer->changeImageLayout(*outputHandler.image, ll::ImageLayout::TransferSrcOptimal);
        m_cmdBuffer->memoryBarrier();
        m_cmdBuffer->copyImageToBuffer(*outputHandler.image, *outputHandler.stagingBuffer);
        m_cmdBuffer->memoryBarrier();
        m_cmdBuffer->changeImageLayout(*outputHandler.image, ll::ImageLayout::General);
        m_cmdBuffer->memoryBarrier();
    }

    m_cmdBuffer->end();

    LOG(INFO) << "LLUVIA: InitNode() finish";

    return ::mediapipe::OkStatus();
}

::mediapipe::Status LluviaCalculator::InitImageFramePort(const ImageFrame* inputImage, PortHandler& portHandler) {

    const auto width = inputImage->Width();
    const auto height = inputImage->Height();

    // TODO: usage flags
    portHandler.stagingBuffer = m_hostMemory->createBuffer(static_cast<uint64_t>(inputImage->PixelDataSize()));
    portHandler.stagingBufferMappedPtr = portHandler.stagingBuffer->map<uint8_t []>();

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

    portHandler.image = m_deviceMemory->createImage(imgDesc);
    portHandler.imageView = portHandler.image->createImageView(ll::ImageViewDescriptor{ll::ImageAddressMode::ClampToBorder,
                                                                                ll::ImageFilterMode::Nearest,
                                                                                false,
                                                                                false});


    portHandler.image->changeImageLayout(ll::ImageLayout::General);

    return ::mediapipe::OkStatus();
}

::mediapipe::Status LluviaCalculator::Process(CalculatorContext* cc) {

    ///////////////////////////////////////////////////////////////////////////
    // init the internals on the first call to Process
    std::call_once(m_configureNode, &LluviaCalculator::InitNode, this, cc);

    ///////////////////////////////////////////////////////////////////////////
    // copy input packets to input handlers
    for (auto& inputHandler : m_inputHandlers) {

        if (inputHandler.mediapipePacketType == lluvia::IMAGE_FRAME) {
            auto& inputImage = cc->Inputs().Tag(inputHandler.mediapipeTag).Get<ImageFrame>();
            inputImage.CopyToBuffer(&inputHandler.stagingBufferMappedPtr[0], inputHandler.stagingBuffer->getSize());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // run the container node
    m_session->run(*m_cmdBuffer);

    ///////////////////////////////////////////////////////////////////////////
    // produce output packets
    for (auto& outputHandler : m_outputHandlers) {

        if (outputHandler.mediapipePacketType == lluvia::IMAGE_FRAME) {

            ::mediapipe::ImageFormat_Format outputImageFormat = ::mediapipe::ImageFormat_Format_UNKNOWN;
            bool imageFormatFound = false;

            std::tie(imageFormatFound, outputImageFormat) =  getMediapipeImageFormat(outputHandler.image->getChannelCount(),
                                                                                     outputHandler.image->getChannelType());
            
            if (!imageFormatFound) {
                return ::mediapipe::UnknownError("unable to find compatible output image format");
            }

            std::unique_ptr<ImageFrame> outputImage = absl::make_unique<ImageFrame>(outputImageFormat,
                                                                                    outputHandler.image->getWidth(),
                                                                                    outputHandler.image->getHeight());

            // copy staging buffer to output ImageFrame
            std::memcpy(outputImage->MutablePixelData(), &outputHandler.stagingBufferMappedPtr[0], outputHandler.stagingBuffer->getSize());

            LOG_EVERY_N(INFO, 300) << "LluviaCalculator: shape [h:"
                                    << std::to_string(outputImage->Height()) << ", w:" << std::to_string(outputImage->Width()) << "], format: "
                                    << std::to_string(static_cast<int>(outputImage->Format())) << ", channel size: "
                                    << std::to_string(outputImage->ChannelSize());

            // TODO: timestamps
            cc->Outputs().Tag(outputHandler.mediapipeTag).Add(outputImage.release(), cc->InputTimestamp());
        }
    }

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
