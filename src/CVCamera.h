/*
Contribution credits:
Leonard Keil <leonard.keil@tum.de>
Jannik Waldraff <jannik.waldraff@tum.de>
Sandro Weber <sandro.weber@tum.de>
*/

#ifndef CV_CAMERA_H
#define CV_CAMERA_H

#include <stdio.h>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/classes/image.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

struct StripDimensions {
    int stripLength;
    int nStop;
    int nStart;
    cv::Point2f stripeVecX; // Vector in edge direction
    cv::Point2f stripeVecY; // Vevtor orhtogonal to VecX
    cv::Size stripSize;     // Dimensions should always be odd numbers
};


namespace godot {

class CVCamera : public RefCounted {
	GDCLASS(CVCamera, RefCounted)

private:
	cv::VideoCapture capture;
    uint64_t last_update_frame;
    cv::Mat frame_raw;
    cv::Mat frame_rgb;
    cv::Mat frame_gray;
    cv::Mat frame_threshold;
    cv::Mat frame_overlay;
    bool flip_lr, flip_ud;
    bool video_file_playback;
    double framerate;

    // Threshold settings
    double threshold_value;
    double threshold_max_value;
    // THRESH_BINARY = 0, THRESH_BINARY_INV = 1, THRESH_TRUNC = 2, THRESH_TOZERO = 3, THRESH_TOZERO_INV = 4, THRESH_MASK = 7,   cv::THRESH_OTSU = 8 , cv::THRESH_TRIANGLE = 16 , cv::THRESH_DRYRUN = 128 
    int threshold_simple_type;
    // cv::ADAPTIVE_THRESH_MEAN_C = 0, cv::ADAPTIVE_THRESH_GAUSSIAN_C = 1
    int threshold_adaptive_type;
    int threshold_blocksize;    // adaptive only
    double threshold_c;     // adaptive only

    // Rectangle detection settings
    int bounding_box_min_width = 20;
    int bounding_box_min_height = 20;
    int bounding_box_max_width = 10;     // As offset from frame size
    int bounding_box_max_height = 10;    // As offset from frame size


    void update_frame();
    Ref<Image> mat_to_image(cv::Mat mat);

    // Helpers
    bool is_valid_threshold_type(int type);

protected:
    static void _bind_methods();

public:
	CVCamera();
	~CVCamera();

    void open(int device, int width, int height);
    void open_file(String path);
    void close();
    Ref<Image> get_image();
    Ref<Image> get_greyscale_image();
    Ref<Image> get_threshold_image();
    Ref<Image> get_overlay_image();
    void find_rectangles();

    void flip(bool flip_lr, bool flip_ud);
    double get_framerate() const;

    // Helper functions
    void calculateStripDimensions(double dx, double dy, StripDimensions &st);
    std::array<cv::Point2f, 4> calculateSubpixCorners(float subpix_line_params[16], bool draw_on_overlay);
    cv::Mat fillStrip(cv::Point2f &center, StripDimensions &st);
    int subpixSampleSafe(const cv::Mat &pSrc, const cv::Point2f &p);
    void calculateSubpixEdgePoint(cv::Point2f subdivision_edge_point, StripDimensions strip_dimensions,cv::Point2f &out_subpix_edge_point);

    // GUI
    void update_threshold_value(double thres_val);
    void update_threshold_max_value(double max_val);
    void update_thres_type(int type);
    void update_threshold_adaptive_type(int type);
    void update_threshold_blocksize(float blocksize);
    void update_threshold_c(double c);

#pragma region Getters&Setters
    int get_bounding_box_min_width() const {
        return bounding_box_min_width;
    }

    void set_bounding_box_min_width(int bounding_box_min_width) {
        this->bounding_box_min_width = bounding_box_min_width;
    }

    int get_bounding_box_min_height() const {
        return bounding_box_min_height;
    }

    void set_bounding_box_min_height(int bounding_box_min_height) {
        this->bounding_box_min_height = bounding_box_min_height;
    }

    int get_bounding_box_max_width() const {
        return bounding_box_max_width;
    }

    void set_bounding_box_max_width(int bounding_box_max_width) {
        this->bounding_box_max_width = bounding_box_max_width;
    }

    int get_bounding_box_max_height() const {
        return bounding_box_max_height;
    }

    void set_bounding_box_max_height(int bounding_box_max_height) {
        this->bounding_box_max_height = bounding_box_max_height;
    }
#pragma endregion
};

} //namespace godot

#endif