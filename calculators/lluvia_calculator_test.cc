
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/formats/image_frame.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

#include "tools/cpp/runfiles/runfiles.h"
using bazel::tools::cpp::runfiles::Runfiles;

#include "lluvia/core.h"

#include <array>
#include <memory>

namespace mediapipe {

namespace {

TEST(LluviaCalculatorTest, TestLoadNodeLibrary) {

    auto runfiles = Runfiles::CreateForTest(nullptr);
    ASSERT_NE(runfiles, nullptr);

    auto libraryPath = runfiles->Rlocation("lluvia/lluvia/nodes/lluvia_node_library.zip");
    LOG(INFO) << "LLUVIA_TEST: library path: " << libraryPath;

    auto session = ll::Session::create(ll::SessionDescriptor().enableDebug(true));

    ASSERT_NO_THROW({
        session->loadLibrary(runfiles->Rlocation(libraryPath));
    });

    auto program = session->getProgram("lluvia/color/RGBA2Gray.comp");
    ASSERT_NE(program, nullptr);

    auto desc = ll::ComputeNodeDescriptor{};
    ASSERT_NO_THROW({
        desc = session->createComputeNodeDescriptor("lluvia/color/RGBA2Gray");
    });

    auto node = std::shared_ptr<ll::ComputeNode>{nullptr};
    ASSERT_NO_THROW({
        node = session->createComputeNode(desc);
    });
    ASSERT_NE(node, nullptr);

    EXPECT_FALSE(session->hasReceivedVulkanWarningMessages());

}

TEST(LluviaCalculatorTest, TestMoni) {

    auto runfiles = Runfiles::CreateForTest(nullptr);
    ASSERT_NE(nullptr, runfiles);

    auto libraryPath = runfiles->Rlocation("lluvia/lluvia/nodes/lluvia_node_library.zip");
    auto lluviaMediapipeLibraryPath = runfiles->Rlocation("mediapipe/mediapipe/lluvia-mediapipe/calculators/lluvia_mediapipe_library.zip");
    
    LOG(INFO) << "LLUVIA_TEST: library path: " << libraryPath;
    LOG(INFO) << "LLUVIA_TEST: library path: " << lluviaMediapipeLibraryPath;


    CalculatorGraphConfig::Node node_config =
        ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
            absl::Substitute(
                R"pb(
                    calculator: "LluviaCalculator"
                    input_stream: "IN_0:input_image"
                    output_stream: "OUT_0:output_image"
                    node_options {
                        [type.googleapis.com/lluvia.LluviaCalculatorOptions]: {
                            enable_debug: true

                            container_node: "lluvia/mediapipe/LluviaCalculator"

                            library_path: "$0"
                            library_path: "$1"

                            input_port_binding:  {
                                mediapipe_tag: "IN_0"
                                lluvia_port: "in_image"
                            }
                        }
                    }
                )pb",
                libraryPath,
                lluviaMediapipeLibraryPath
            )
        );
    
    CalculatorRunner runner(node_config);

    // cv::Mat input_mat;
    // cv::cvtColor(cv::imread(file::JoinPath("./",
    //                                         "/mediapipe/calculators/"
    //                                         "image/testdata/dino.jpg")),
    //             input_mat, cv::COLOR_BGR2RGB);
    Packet input_packet = MakePacket<ImageFrame>(ImageFormat::SRGBA, 1920, 1080);
    // input_mat.copyTo(formats::MatView(&(input_packet.Get<ImageFrame>())));

    runner.MutableInputs()->Tag("IN_0").packets.push_back(input_packet.At(Timestamp(0)));


    MP_ASSERT_OK(runner.Run());
}


TEST(LluviaCalculatorTest, TestCompatibleImageFormats) {

    auto runfiles = Runfiles::CreateForTest(nullptr);
    ASSERT_NE(nullptr, runfiles);

    auto libraryPath = runfiles->Rlocation("lluvia/lluvia/nodes/lluvia_node_library.zip");
    auto calculatorScriptPath = runfiles->Rlocation("mediapipe/mediapipe/lluvia-mediapipe/calculators/test_data/PassthroughContainerNode.lua");
    
    LOG(INFO) << "LLUVIA_TEST: library path: " << libraryPath;
    LOG(INFO) << "LLUVIA_TEST: script path: " << calculatorScriptPath;


    CalculatorGraphConfig::Node node_config =
        ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
            absl::Substitute(
                R"pb(
                    calculator: "LluviaCalculator"
                    input_stream: "IN_0:input_image"
                    output_stream: "OUT_0:output_image"
                    node_options {
                        [type.googleapis.com/lluvia.LluviaCalculatorOptions]: {
                            enable_debug: true

                            container_node: "mediapipe/test/PassthroughContainerNode"

                            library_path: "$0"

                            script_path: "$1"

                            input_port_binding:  {
                                mediapipe_tag: "IN_0"
                                lluvia_port: "in_image"
                            }
                        }
                    }
                )pb",
                libraryPath,
                calculatorScriptPath
            )
        );
    
    // FIXME: only these formats work, all others complain about ByteDepth different to 1 image_frame.cc:379
    const auto imageFormats = std::array {ImageFormat::SRGBA, ImageFormat::GRAY8};

    
    for (const auto& imageFormat : imageFormats) {

        CalculatorRunner runner(node_config);

        Packet input_packet = MakePacket<ImageFrame>(imageFormat, 1920, 1080);

        runner.MutableInputs()->Tag("IN_0").packets.push_back(input_packet.At(Timestamp(0)));

        MP_ASSERT_OK(runner.Run());

        LOG(INFO) << "packet size: " << runner.Outputs().Tag("OUT_0").packets.size();

        ASSERT_TRUE(runner.Outputs().Tag("OUT_0").packets.size() >= 1);

        auto outPacket = runner.Outputs().Tag("OUT_0").packets[0];

        auto& out_image = outPacket.Get<ImageFrame>();

        ASSERT_EQ(out_image.Format(), imageFormat);
        ASSERT_EQ(out_image.Width(), 1920);
        ASSERT_EQ(out_image.Height(), 1080);
    }
    
    
}


} // namespace
} // namespace mediapipe
