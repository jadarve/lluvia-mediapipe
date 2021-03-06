// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#include <lluvia/core.h>
#include <vulkan/vulkan.hpp>

#include <memory>
#include <mutex>

#define HAVE_GPU_BUFFER
#ifdef __APPLE__
#include "mediapipe/objc/util.h"
#endif

#include "mediapipe/gpu/gl_calculator_helper.h"

namespace mediapipe {

/**
 * A Deleter that does nothing :)
 */
struct NopDeleter{

    template<typename T>
    void operator ()(T* ptr) const {
    }
};

// Convert an input image (GpuBuffer or ImageFrame) to ImageFrame.
class LluviaFromGPUBufferCalculator : public CalculatorBase {

    public:
        LluviaFromGPUBufferCalculator() = default;

        static ::mediapipe::Status GetContract(CalculatorContract* cc);

        ::mediapipe::Status Open(CalculatorContext* cc) override;
        ::mediapipe::Status Process(CalculatorContext* cc) override;

    private:
        void InitNode(const mediapipe::GpuBuffer*);

    private:
        std::shared_ptr<ll::Session> m_session {};
        std::shared_ptr<ll::Memory> m_deviceLocalMemory {};
        std::shared_ptr<ll::Memory> m_stagingMemory {};

        std::unique_ptr<ll::CommandBuffer> m_cmdBuffer {};

        std::shared_ptr<ll::ComputeNode> m_computeNode {};

        std::shared_ptr<ll::Buffer> m_inputStagingBuffer {};
        std::unique_ptr<uint8_t[], ll::Buffer::BufferMapDeleter> m_inputStagingBufferMapped {};
        std::shared_ptr<ll::Image> m_inputImage {};
        std::shared_ptr<ll::ImageView> m_inputImageView {};

        std::mutex m_outputStagingBufferLock {};
        std::shared_ptr<ll::Buffer> m_outputStagingBuffer {};
        std::unique_ptr<uint8_t[], ll::Buffer::BufferMapDeleter> m_outputStagingBufferMapped {};
        std::shared_ptr<ll::Image> m_outputImage {};
        std::shared_ptr<ll::ImageView> m_outputImageView {};

        std::once_flag m_configureNode {};

#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
        GlCalculatorHelper helper_;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    };
    REGISTER_CALCULATOR(LluviaFromGPUBufferCalculator);


    ::mediapipe::Status LluviaFromGPUBufferCalculator::GetContract(
            CalculatorContract* cc) {
        cc->Inputs().Index(0).Set<GpuBuffer>();
        cc->Outputs().Index(0).Set<GpuBuffer>();
        // Note: we call this method even on platforms where we don't use the helper,
        // to ensure the calculator's contract is the same. In particular, the helper
        // enables support for the legacy side packet, which several graphs still use.
        MP_RETURN_IF_ERROR(GlCalculatorHelper::UpdateContract(cc));
        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status LluviaFromGPUBufferCalculator::Open(CalculatorContext* cc) {

        // Inform the framework that we always output at the same timestamp
        // as we receive a packet at.
        cc->SetOffset(TimestampDiff(0));
#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
        MP_RETURN_IF_ERROR(helper_.Open(cc));
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

        auto sessionDesc = ll::SessionDescriptor{}
                               .enableDebug(true);

        m_session = ll::Session::create(sessionDesc);

        auto memoryProperties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        m_stagingMemory = m_session->createMemory(memoryProperties, 0, false);
        m_deviceLocalMemory = m_session->createMemory(vk::MemoryPropertyFlagBits::eDeviceLocal, 32 * 1024 * 1024, false);


        LOG(INFO) << "LluviaCalculator: memories created: statingMemory: " << m_stagingMemory->getPageSize() << " deviceLocalMemory: " << m_deviceLocalMemory->getPageSize();

        // path to the .zip containing all the nodes
        auto string_path = std::string {};
        ASSIGN_OR_RETURN(string_path, mediapipe::PathToResourceAsFile("lluvia_node_library.zip"));

        m_session->loadLibrary(string_path);

        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status LluviaFromGPUBufferCalculator::Process(CalculatorContext* cc) {

        auto guard = std::lock_guard {m_outputStagingBufferLock};

#ifdef HAVE_GPU_BUFFER
        if (cc->Inputs().Index(0).Value().ValidateAsType<GpuBuffer>().ok()) {

            const auto& input = cc->Inputs().Index(0).Get<GpuBuffer>();

            // init the internals given a concrete GPUBuffer
            std::call_once(m_configureNode, &LluviaFromGPUBufferCalculator::InitNode, this, &input);

            // transfer the GPUBuffer pixel content to the input stagging buffer (already mapped)
            helper_.RunInGlContext([this, &input, &cc]() {

                auto src = helper_.CreateSourceTexture(input);

                helper_.BindFramebuffer(src);
                const auto info = GlTextureInfoForGpuBufferFormat(input.format(), 0);
                glReadPixels(0, 0, src.width(), src.height(), info.gl_format,
                             info.gl_type, &this->m_inputStagingBufferMapped[0]);

                glFlush();
                src.Release();
            });
        }
#endif  // defined(HAVE_GPU_BUFFER)

        // Run the Lluvia graph
        m_session->run(*m_cmdBuffer);


        // transfer the pixels from outputImage to GPUBuffer
        helper_.RunInGlContext([this, &cc]() {

            std::unique_ptr<ImageFrame> outputImage = absl::make_unique<ImageFrame>(
                    ImageFormat::GRAY8,
                    this->m_outputImage->getWidth(),
                    this->m_outputImage->getHeight(),
                    this->m_outputImage->getSize() / this->m_outputImage->getHeight(),
                    &(this->m_outputStagingBufferMapped[0]),
                    NopDeleter{}
                    );

            auto src = this->helper_.CreateSourceTexture(*outputImage);
            auto output = src.GetFrame<GpuBuffer>();
            glFlush();

            // FIXME timestamp
            cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
            src.Release();

            LOG_EVERY_N(INFO, 300) << "LluviaCalculator: shape [h:"
                                   << std::to_string(outputImage->Height()) << ", w:" << std::to_string(outputImage->Width()) << "], format: "
                                   << std::to_string(static_cast<int>(outputImage->Format())) << ", channel size: "
                                   << std::to_string(outputImage->ChannelSize());
        });

        return ::mediapipe::OkStatus();
    }

    void LluviaFromGPUBufferCalculator::InitNode(const mediapipe::GpuBuffer* inputImage) {

        const auto width = inputImage->width();
        const auto height = inputImage->height();

        LOG(INFO) << "LluviaCalculator: InitNode() width: " << width << " height: " << height;

        ///////////////////////////////////////////////////
        // Input image
        ///////////////////////////////////////////////////

        // Create an ImageFrame in order to know the raw byte size I need to
        // allocate in the staging buffer
        std::unique_ptr<ImageFrame> inputFrame = absl::make_unique<ImageFrame>(
                ImageFormatForGpuBufferFormat(inputImage->format()), width,
                height, ImageFrame::kGlDefaultAlignmentBoundary);

        m_inputStagingBuffer = m_stagingMemory->createBuffer(inputFrame->PixelDataSizeStoredContiguously());

        // permanently map the staging buffers
        m_inputStagingBufferMapped = m_inputStagingBuffer->map<uint8[]>();

        LOG(INFO) << "LluviaCalculator: InitNode() inputStagingBuffer  : " << m_inputStagingBuffer->getAllocationInfo().page << " size: " << m_inputStagingBuffer->getSize();

        ///////////////////////////////////////////////////
        // Compute node
        ///////////////////////////////////////////////////

        // FIXME: the name of the node should be passed as parameter
        m_computeNode =  m_session->createComputeNode("lluvia/RGBA2Gray");


        const vk::ImageUsageFlags imgUsageFlags = { vk::ImageUsageFlagBits::eStorage
                                                    | vk::ImageUsageFlagBits::eSampled
                                                    | vk::ImageUsageFlagBits::eTransferDst
                                                    | vk::ImageUsageFlagBits::eTransferSrc};

        const auto imgDesc = ll::ImageDescriptor{1, static_cast<uint32_t >(height), static_cast<uint32_t>(width),
                                                 ll::ChannelCount::C4, ll::ChannelType::Uint8}
                                                 .setUsageFlags(imgUsageFlags);

        m_inputImage = m_deviceLocalMemory->createImage(imgDesc);
        m_inputImageView = m_inputImage->createImageView(ll::ImageViewDescriptor{ll::ImageAddressMode::ClampToBorder,
                                                                                 ll::ImageFilterMode::Nearest,
                                                                                 false,
                                                                                 false});


        m_inputImage->changeImageLayout(vk::ImageLayout::eGeneral);

        // FIXME: input name should be passed as parameter
        m_computeNode->bind("in_rgba", m_inputImageView);
        m_computeNode->init();


        ///////////////////////////////////////////////////
        // Output image
        ///////////////////////////////////////////////////

        // FIXME: output name should be passed as parameter
        m_outputImageView = std::static_pointer_cast<ll::ImageView>(m_computeNode->getPort("out_gray"));
        m_outputImage = m_outputImageView->getImage();

        std::unique_ptr<ImageFrame> outputFrame = absl::make_unique<ImageFrame>(
                ImageFormat_Format_GRAY8, width,
                height, ImageFrame::kGlDefaultAlignmentBoundary);

        m_outputStagingBuffer = m_stagingMemory->createBuffer(outputFrame->PixelDataSizeStoredContiguously());
        m_outputStagingBufferMapped = m_outputStagingBuffer->map<uint8_t[]>();

        LOG(INFO) << "LluviaCalculator: InitNode() outputStagingBuffer : " << m_outputStagingBuffer->getAllocationInfo().page << " size: " << m_outputStagingBuffer->getSize();


        ///////////////////////////////////////////////////
        // Command buffer
        ///////////////////////////////////////////////////

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

        LOG(INFO) << "LluviaCalculator: InitNode() finish";
    }

}  // namespace mediapipe
