#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#include <lluvia/core.h>
#include <vulkan/vulkan.hpp>

#include <memory>

namespace mediapipe {

// Calculator to pass a CPU image through. It prints in the logs the
// image attributes such as resolution and format.
    class LluviaCalculator : public CalculatorBase {
    public:
        static ::mediapipe::Status GetContract(CalculatorContract* cc);
        ::mediapipe::Status Open(CalculatorContext* cc) override;
        ::mediapipe::Status Process(CalculatorContext* cc) override;
        ::mediapipe::Status Close(CalculatorContext* cc) override;

        void InitNode(const ImageFrame*);

    private:
        std::shared_ptr<ll::Session> m_session {};
        std::shared_ptr<ll::Memory> m_memory {};

        std::unique_ptr<ll::CommandBuffer> m_cmdBuffer {};

        std::shared_ptr<ll::ComputeNode> m_computeNode {};

        std::shared_ptr<ll::Buffer> m_inputStagingBuffer {};
        std::shared_ptr<ll::Image> m_inputImage {};
        std::shared_ptr<ll::ImageView> m_inputImageView {};

        std::shared_ptr<ll::Buffer> m_outputStagingBuffer {};
        std::shared_ptr<ll::Image> m_outputImage {};
        std::shared_ptr<ll::ImageView> m_outputImageView {};


        std::once_flag m_configureNode {};
    };

    ::mediapipe::Status LluviaCalculator::GetContract(
            CalculatorContract* cc) {

        // FIXME: try to support GPUBuffer instead :)
        cc->Inputs().Index(0).Set<ImageFrame>();
        cc->Outputs().Index(0).Set<ImageFrame>();
        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status LluviaCalculator::Open(CalculatorContext* cc) {

        m_session = ll::Session::create();

        // FIXME: memory properties depend on the platform
        auto memoryProperties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        m_memory = m_session->createMemory(memoryProperties, 32 * 1024 * 1024, false);

        LOG(INFO) << "JUAN: memory created: " << m_memory->getPageSize();

        // open a file. It works!!!
        auto string_path = std::string {};
        ASSIGN_OR_RETURN(string_path, mediapipe::PathToResourceAsFile("lluvia_node_library.zip"));
        LOG(INFO) << "JUAN: file path: " << string_path;

        m_session->loadLibrary(string_path);

        auto program = m_session->getProgram("lluvia/RGBA2Gray");
        LOG(INFO) << "JUAN: program found: " << (program != nullptr);

        auto desc = m_session->createComputeNodeDescriptor("lluvia/RGBA2Gray");
        LOG(INFO) << "JUAN: desc: " << desc.getBuilderName();

        return ::mediapipe::OkStatus();
    }

    void LluviaCalculator::InitNode(const ImageFrame* inputImage) {

        LOG(INFO) << "JUAN: InitNode() start";

        const auto width = inputImage->Width();
        const auto height = inputImage->Height();

        LOG(INFO) << "JUAN: InitNode() width: " << width << " height: " << height;

        m_computeNode =  m_session->createComputeNode("lluvia/RGBA2Gray");

        // TODO: usage flags
        m_inputStagingBuffer = m_memory->createBuffer(static_cast<uint64_t>(inputImage->PixelDataSize()));

        const vk::ImageUsageFlags imgUsageFlags = { vk::ImageUsageFlagBits::eStorage
                                                    | vk::ImageUsageFlagBits::eSampled
                                                    | vk::ImageUsageFlagBits::eTransferDst
                                                    | vk::ImageUsageFlagBits::eTransferSrc};

        const auto imgDesc = ll::ImageDescriptor{1, static_cast<uint32_t >(height), static_cast<uint32_t>(width),
                                                 ll::ChannelCount::C4, ll::ChannelType::Uint8}
                 .setUsageFlags(imgUsageFlags);

        m_inputImage = m_memory->createImage(imgDesc);
        m_inputImageView = m_inputImage->createImageView(ll::ImageViewDescriptor{ll::ImageAddressMode::ClampToBorder,
                                                                                 ll::ImageFilterMode::Nearest,
                                                                                 false,
                                                                                 false});


        m_inputImage->changeImageLayout(vk::ImageLayout::eGeneral);

        m_computeNode->bind("in_rgba", m_inputImageView);
        m_computeNode->init();

        m_outputImageView = std::static_pointer_cast<ll::ImageView>(m_computeNode->getPort("out_gray"));
        m_outputImage = m_outputImageView->getImage();
        // FIXME: what's the correct size?
        m_outputStagingBuffer = m_memory->createBuffer(static_cast<uint64_t>(width * height));


        // Create command buffer
        m_cmdBuffer = m_session->createCommandBuffer();
        m_cmdBuffer->begin();

        // Copy staging buffer to m_inputImage. Consider changing image layout.
        m_cmdBuffer->changeImageLayout(*m_inputImage, vk::ImageLayout::eTransferDstOptimal);
        m_cmdBuffer->memoryBarrier();
        m_cmdBuffer->copyBufferToImage(*m_inputStagingBuffer, *m_inputImage);
        m_cmdBuffer->memoryBarrier(); // TODO: needed?
        m_cmdBuffer->changeImageLayout(*m_inputImage, vk::ImageLayout::eGeneral);
        m_cmdBuffer->memoryBarrier();

        // Compute
        m_cmdBuffer->run(*m_computeNode);
        m_cmdBuffer->memoryBarrier();

        // Copy output image to staging buffer
        m_cmdBuffer->changeImageLayout(*m_outputImage, vk::ImageLayout::eTransferSrcOptimal);
        m_cmdBuffer->memoryBarrier();
        m_cmdBuffer->copyImageToBuffer(*m_outputImage, *m_outputStagingBuffer);
        m_cmdBuffer->memoryBarrier();
        m_cmdBuffer->changeImageLayout(*m_outputImage, vk::ImageLayout::eGeneral);

        m_cmdBuffer->end();

        LOG(INFO) << "JUAN: InitNode() finish";
    }

    ::mediapipe::Status LluviaCalculator::Process(CalculatorContext* cc) {


        auto& inputImage = cc->Inputs().Index(0).Get<ImageFrame>();

        // init the internals given a concrete ImageFrame
        std::call_once(m_configureNode, &LluviaCalculator::InitNode, this, &inputImage);

        // Copy input image to a staging buffer
        {
            // TODO: add another overload of mapAndSet
            auto ptr = m_inputStagingBuffer->map<uint8_t []>();
            inputImage.CopyToBuffer(&ptr[0], m_inputStagingBuffer->getSize());
        }

        m_session->run(*m_cmdBuffer);

        std::unique_ptr<ImageFrame> outputImage = absl::make_unique<ImageFrame>(
                ImageFormat::GRAY8, inputImage.Width(), inputImage.Height());

        // copy staging buffer to output ImageFrame
        {
            auto ptr = m_outputStagingBuffer->map<uint8_t[]>();
            std::memcpy(outputImage->MutablePixelData(), &ptr[0], m_outputStagingBuffer->getSize());
        }

        LOG_EVERY_N(INFO, 300) << "LluviaCalculator: shape [h:"
                               << std::to_string(outputImage->Height()) << ", w:" << std::to_string(outputImage->Width()) << "], format: "
                               << std::to_string(static_cast<int>(outputImage->Format())) << ", channel size: "
                               << std::to_string(outputImage->ChannelSize());

        cc->Outputs().Index(0).Add(outputImage.release(), cc->InputTimestamp());


        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status LluviaCalculator::Close(CalculatorContext* cc) {
        return ::mediapipe::OkStatus();
    }

    REGISTER_CALCULATOR(LluviaCalculator);

}  // namespace mediapipe
