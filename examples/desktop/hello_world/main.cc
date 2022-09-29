
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

#include "tools/cpp/runfiles/runfiles.h"
using bazel::tools::cpp::runfiles::Runfiles;

#include <iostream>

ABSL_FLAG(std::string, input_image, "", "Path to the input image.");
ABSL_FLAG(std::string, script_file, "", "Path to the LUA script describing the container node.");

ABSL_FLAG(std::string, graph_file, "", "Name of file containing text format CalculatorGraphConfig proto.");


absl::Status runGraph(const std::string mainFileLocation) {

    auto input_image = absl::GetFlag(FLAGS_input_image);
    if (input_image.empty()) {
        return absl::InvalidArgumentError("input_image cannot be empty");
    }

    auto script_file = absl::GetFlag(FLAGS_script_file);
    if (script_file.empty()) {
        return absl::InvalidArgumentError("script_file cannot be empty");
    }

    auto graph_file = absl::GetFlag(FLAGS_graph_file);
    if (graph_file.empty()) {
        return absl::InvalidArgumentError("graph_file cannot be empty");
    }

    ///////////////////////////////////////////////////////////////////////////
    // Load node library
    auto runfiles = Runfiles::Create(mainFileLocation);
    auto libraryPath = runfiles->Rlocation("lluvia/lluvia/nodes/lluvia_node_library.zip");    

    ///////////////////////////////////////////////////////////////////////////
    // Graph configuration
    auto graphConfigFileContent = std::string {};
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(absl::GetFlag(FLAGS_graph_file), &graphConfigFileContent));

    // replace template values
    graphConfigFileContent = absl::Substitute(graphConfigFileContent, libraryPath, script_file);

    mediapipe::CalculatorGraphConfig graphConfig = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graphConfigFileContent);

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(graphConfig));

    ///////////////////////////////////////////////////////////////////////////
    // Run the graph
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller outputPoller, graph.AddOutputStreamPoller("output_stream"));
    MP_RETURN_IF_ERROR(graph.StartRun({}));
    

    ///////////////////////////////////////////////////////////////////////////
    // Read input image
    auto cvInputImage = cv::imread(input_image);
    cv::cvtColor(cvInputImage, cvInputImage, cv::COLOR_BGR2RGBA);

    
    ///////////////////////////////////////////////////////////////////////////
    // Feed the graph
    auto inputPacket = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGBA, cvInputImage.cols, cvInputImage.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat inputPacketMat = mediapipe::formats::MatView(inputPacket.get());
    cvInputImage.copyTo(inputPacketMat);

    MP_RETURN_IF_ERROR(
        graph.AddPacketToInputStream("input_stream", 
            mediapipe::Adopt(inputPacket.release()).At(mediapipe::Timestamp(0))
        )
    );

    ///////////////////////////////////////////////////////////////////////////
    // Get the output
    mediapipe::Packet outputPacket;

    if (!outputPoller.Next(&outputPacket)) {
        return absl::UnknownError("No package to poll");
    }

    auto& outputImageFrame = outputPacket.Get<mediapipe::ImageFrame>();
    auto outputImageMat = mediapipe::formats::MatView(&outputImageFrame);
    
    auto cvOutputImage = cv::Mat {};
    outputImageMat.copyTo(cvOutputImage);

    cv::cvtColor(cvOutputImage, cvOutputImage, cv::COLOR_RGBA2BGR);


    ///////////////////////////////////////////////////////////////////////////
    // Render
    cv::imshow("input_image", cvInputImage);
    cv::imshow("output_image", cvOutputImage);
    cv::waitKey(0);

    return absl::OkStatus();

}


/*
bazel run --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
    //mediapipe/lluvia-mediapipe/examples/desktop/hello_world:hello_world -- \
    --input_image=${HOME}/git/lluvia/lluvia/resources/mouse.jpg \
    --script_file=${HOME}/git/mediapipe/mediapipe/lluvia-mediapipe/examples/desktop/hello_world/script.lua \
    --graph_file=${HOME}/git/mediapipe/mediapipe/lluvia-mediapipe/examples/desktop/hello_world/graph.pbtxt

*/
int main(int argc, char** argv) {

    ///////////////////////////////////////////////////////////////////////////
    // Arg parsing
    absl::ParseCommandLine(argc, argv);
    std::cout << "input_image: " << absl::GetFlag(FLAGS_input_image) << std::endl;
    std::cout << "script_file: " << absl::GetFlag(FLAGS_script_file) << std::endl;

    auto status = runGraph(std::string {argv[0]});
    
    if (!status.ok()) {
        std::cerr << "ERROR: " << status.message() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
