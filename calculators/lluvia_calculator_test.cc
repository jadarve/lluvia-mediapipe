
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
    LOG(INFO) << "LLUVIA_TEST: library path: " << libraryPath;

    CalculatorGraphConfig::Node node_config =
        ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
            absl::Substitute(
                R"pb(
                    calculator: "LluviaCalculator"
                    input_stream: "input_image"
                    output_stream: "output_image"
                    node_options {
                        [type.googleapis.com/lluvia.LluviaCalculatorOptions]: {
                            libraryPath: "$0"
                        }
                    }
                )pb",
                libraryPath
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

    runner.MutableInputs()->Index(0).packets.push_back(input_packet.At(Timestamp(0)));


    MP_ASSERT_OK(runner.Run());
}


} // namespace
} // namespace mediapipe
