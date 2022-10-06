
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

ABSL_FLAG(std::string, script_file, "", "Path to the LUA script describing the container node.");

ABSL_FLAG(std::string, graph_file, "", "Name of file containing text format CalculatorGraphConfig proto.");


absl::Status runGraph(const std::string mainFileLocation) {

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
    // Open the video capture device
    auto videoCapture = cv::VideoCapture {};
    videoCapture.open(0, cv::CAP_ANY);

    if (!videoCapture.isOpened()) {
        return absl::UnknownError("Error opening capture device.");
    }

    auto cvInputImage = cv::Mat {};
    auto cvInputImageBGRA = cv::Mat {};

    auto timestampCounter = int64_t {0};
    for (;;) {   

        ///////////////////////////////////////////////////////////////////////////
        // Read input image
        videoCapture.read(cvInputImage);

        if (cvInputImage.empty()) {
            return absl::UnknownError("Error reading image from capture device.");
        }

        cv::cvtColor(cvInputImage, cvInputImageBGRA, cv::COLOR_BGR2BGRA);

        
        ///////////////////////////////////////////////////////////////////////////
        // Feed the graph
        auto inputPacket = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SBGRA, cvInputImageBGRA.cols, cvInputImageBGRA.rows,
            mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
        cv::Mat inputPacketMat = mediapipe::formats::MatView(inputPacket.get());
        cvInputImageBGRA.copyTo(inputPacketMat);

        MP_RETURN_IF_ERROR(
            graph.AddPacketToInputStream("input_stream", 
                mediapipe::Adopt(inputPacket.release()).At(mediapipe::Timestamp(timestampCounter++))
            )
        );

        ///////////////////////////////////////////////////////////////////////////
        // Get the output
        mediapipe::Packet outputPacket;

        if (!outputPoller.Next(&outputPacket)) {
            // no package to poll in this loop iteration
            return absl::UnknownError("Error polling output packet");
        }

        auto& outputImageFrame = outputPacket.Get<mediapipe::ImageFrame>();
        auto outputImageMat = mediapipe::formats::MatView(&outputImageFrame);
        
        auto cvOutputImage = cv::Mat {};
        outputImageMat.copyTo(cvOutputImage);

        ///////////////////////////////////////////////////////////////////////////
        // Render
        cv::imshow("input_image", cvInputImage);
        cv::imshow("output_image", cvOutputImage);
        
        if (cv::waitKey(40) >= 0) {
            break;
        }

    }

    return absl::OkStatus();

}


int main(int argc, char** argv) {

    ///////////////////////////////////////////////////////////////////////////
    // Arg parsing
    absl::ParseCommandLine(argc, argv);
    std::cout << "script_file: " << absl::GetFlag(FLAGS_script_file) << std::endl;

    auto status = runGraph(std::string {argv[0]});
    
    if (!status.ok()) {
        std::cerr << "ERROR: " << status.message() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
