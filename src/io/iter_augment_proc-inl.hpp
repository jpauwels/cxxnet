#ifndef CXXNET_ITER_AUGMENT_INL_HPP_
#define CXXNET_ITER_AUGMENT_INL_HPP_
/*!
 * \file iter_augment_proc-inl.hpp
 * \brief processing unit to do data augmention
 * \author Tianqi Chen, Bing Xu, Naiyan Wang
 */
#include <sstream>
#include <mshadow/tensor.h>
#include "data.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../utils/global_random.h"
#include "../utils/thread_buffer.h"

#if CXXNET_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace cxxnet {
/*! \brief create a batch iterator from single instance iterator */
class AugmentIterator: public IIterator<DataInst> {
public:
  AugmentIterator(IIterator<DataInst> *base): base_(base) {
    rand_crop_ = 0;
    rand_mirror_ = 0;
    // skip read, used for debug
    test_skipread_ = 0;
    // scale data
    scale_ = 1.0f;
    // number of overflow instances that readed in round_batch mode
    num_overflow_ = 0;
    // silent
    silent_ = 0;
    // by default, not mean image file
    name_meanimg_ = "";
    crop_y_start_ = -1;
    crop_x_start_ = -1;
    max_rotate_angle_ = 0.0f;
    max_aspect_ratio_ = 0.0f;
    max_shear_ratio_ = 0.0f;
    min_crop_size_ = -1;
    max_crop_size_ = -1;
    mirror_ = 0;
    rotate_ = -1.0f;
    resize_ = "none";
  }
  virtual ~AugmentIterator(void) {
    delete base_;
  }
  virtual void SetParam(const char *name, const char *val) {
    base_->SetParam(name, val);
    if (!strcmp(name, "input_shape")) {
      utils::Assert(sscanf(val, "%u,%u,%u", &shape_[0], &shape_[1], &shape_[2]) == 3,
                    "input_shape must be three consecutive integers without space example: 1,1,200 ");
    }
    if (!strcmp(name, "rand_crop"))   rand_crop_ = atoi(val);
    if (!strcmp(name, "crop_y_start"))  crop_y_start_ = atoi(val);
    if (!strcmp(name, "crop_x_start"))  crop_x_start_ = atoi(val);
    if (!strcmp(name, "rand_mirror")) rand_mirror_ = atoi(val);
    if (!strcmp(name, "silent"))      silent_ = atoi(val);
    if (!strcmp(name, "divideby"))    scale_ = static_cast<real_t>(1.0f / atof(val));
    if (!strcmp(name, "scale"))       scale_ = static_cast<real_t>(atof(val));
    if (!strcmp(name, "image_mean"))   name_meanimg_ = val;
    if (!strcmp(name, "max_rotate_angle")) max_rotate_angle_ = atof(val);
    if (!strcmp(name, "max_shear_ratio"))  max_shear_ratio_ = atof(val);
    if (!strcmp(name, "max_aspect_ratio"))  max_aspect_ratio_ = atof(val);
    if (!strcmp(name, "test_skipread"))    test_skipread_ = atoi(val);
    if (!strcmp(name, "min_crop_size"))     min_crop_size_ = atoi(val);
    if (!strcmp(name, "max_crop_size"))     max_crop_size_ = atoi(val);
    if (!strcmp(name, "mirror")) mirror_ = atoi(val);
    if (!strcmp(name, "rotate")) rotate_ = atoi(val);
    if (!strcmp(name, "rotate_list")) {
      const char *end = val + strlen(val);
      char buf[128];
      while (val < end) {
        sscanf(val, "%[^,]", buf);
        val += strlen(buf) + 1;
        rotate_list_.push_back(atoi(buf));
      }
    }
    if (!strcmp(name, "mean_value")) {
      std::istringstream meanStream(val);
      float mean;
      while (meanStream >> mean) {
        means_.push_back(mean);
        meanStream.ignore(1, ',');
      }
      utils::Assert(means_.size() > 0,
                    "mean_value must be one or more floats, separated by commas, e.g. '128,127.5,128.2'");
    }
    if (!strcmp(name, "resize")) resize_ = val;
  }
  virtual void Init(void) {
    base_->Init();
    printf("In augment init.\n");
    meanfile_ready_ = false;
    if (name_meanimg_.length() != 0) {
      FILE *fi = fopen64(name_meanimg_.c_str(), "rb");
      if (fi == NULL) {
        this->CreateMeanImg();
      } else {
        if (silent_ == 0) {
          printf("loading mean image from %s\n", name_meanimg_.c_str());
        }
        utils::FileStream fs(fi) ;
        meanimg_.LoadBinary(fs);
        fclose(fi);
      }
      #if CXXNET_USE_OPENCV
      // convert mean image to cv::Mat
      cv::Mat floatImg;
      if (meanimg_.size(0) == 1) {
        floatImg = cv::Mat(meanimg_.size(1), meanimg_.size(2), CV_32FC1, meanimg_.dptr_);
      } else {
        cv::Mat r_pixels(meanimg_.size(1), meanimg_.size(2), CV_32FC1, &meanimg_[0][0][0]);
        cv::Mat g_pixels(meanimg_.size(1), meanimg_.size(2), CV_32FC1, &meanimg_[1][0][0]);
        cv::Mat b_pixels(meanimg_.size(1), meanimg_.size(2), CV_32FC1, &meanimg_[2][0][0]);
        cv::Mat reordered_pixels[] = {b_pixels, g_pixels, r_pixels};
        cv::merge(reordered_pixels, 3, floatImg);
      }
      floatImg.convertTo(meanOpenCVImage_, CV_8U);
      #endif
      meanfile_ready_ = true;
    }
  }
  virtual void BeforeFirst(void) {
    base_->BeforeFirst();
  }
  virtual bool Next(void) {
    if (!base_->Next()){
      return false;
    }
    const DataInst &d = base_->Value();
    this->SetData(d);
    return true;
  }
  virtual const DataInst &Value(void) const {
    return out_;
  }
private:
  inline void SetData(const DataInst &d) {
    using namespace mshadow::expr;
    out_.label = d.label;
    out_.index = d.index;
    utils::Assert(d.data.size(0) == shape_[0],
                  "The number of channels in the image should match the number of network inputs");
    utils::Assert(!meanfile_ready_ || d.data.size(0) == meanimg_.size(0),
                  "The number of channels in the mean image should be the same as in the image");
    utils::Assert(means_.size() == 0 || means_.size() == d.data.size(0),
                  "The dimensions of the mean value should match the number of image channels");
    img_.Resize(mshadow::Shape3(shape_[0], shape_[1], shape_[2]));
    #if CXXNET_USE_OPENCV
    cv::Mat res(d.data.size(1), d.data.size(2), CV_8UC(d.data.size(0)), d.data.dptr_, d.data.stride_);
    bool meanSubtracted = false;
    if (shape_[1] > 1) {
      // Resizing
      if (resize_ == "none") {
      } else if (resize_ == "hard") { // breaks aspect ratio
        cv::resize(res, res, cv::Size(shape_[1], shape_[2]));
      } else if (resize_ == "area") { // flattens
      } else {
        float imageAspectRatio = static_cast<float>(res.cols) / res.rows;
        float networkAspectRatio = static_cast<float>(shape_[2]) / shape_[1];
        cv::Rect roi;
        if (resize_ == "inner") { // preserves aspect ratio
          if (imageAspectRatio > networkAspectRatio) {
            int resizeWidth = cvRound(res.rows * networkAspectRatio);
            roi = cv::Rect((res.cols - resizeWidth)/2, 0, resizeWidth, res.rows);
          } else {
            int resizeHeight = cvRound(res.cols * networkAspectRatio);
            roi = cv::Rect(0, (res.rows - resizeHeight)/2, res.cols, resizeHeight);
          }
        } else if (resize_ == "outer") { // preserves aspect ratio
          
        }
        res = res(roi);
        cv::resize(res, res, cv::Size(shape_[1], shape_[2]));
      }
      // Affine transformation
      if (max_rotate_angle_ > 0 || max_shear_ratio_ > 0.0f
          || rotate_ > 0 || rotate_list_.size() > 0) {
        int angle = utils::NextUInt32(max_rotate_angle_ * 2) - max_rotate_angle_;
        if (rotate_ > 0) angle = rotate_;
        if (rotate_list_.size() > 0) angle = rotate_list_[utils::NextUInt32(rotate_list_.size() - 1)];
        int len = std::max(res.cols, res.rows);
        cv::Mat M(2, 3, CV_32F);
        float cs = cos(angle / 180.0 * M_PI);
        float sn = sin(angle / 180.0 * M_PI);
        float q = utils::NextDouble() * max_shear_ratio_ * 2 - max_shear_ratio_;
        M.at<float>(0, 0) = cs;
        M.at<float>(0, 1) = sn;
        M.at<float>(0, 2) = (1 - cs - sn) * len / 2.0;
        M.at<float>(1, 0) = q * cs - sn;
        M.at<float>(1, 1) = q * sn + cs;
        M.at<float>(1, 2) = (1 - cs + sn) * len / 2.0;
        cv::Mat temp;
        cv::warpAffine(res, temp, M, cv::Size(len, len),
              cv::INTER_CUBIC,
              cv::BORDER_CONSTANT,
              cv::Scalar(255, 255, 255));
        res = temp;
      }
      // Subtracting mean image
      if (res.rows == meanOpenCVImage_.rows && res.cols == meanOpenCVImage_.cols && meanfile_ready_) {
        res -= meanOpenCVImage_;
        meanSubtracted = true;
      }
      // Cropping
      int crop_width;
      int crop_height;
      if (min_crop_size_ > 0 && max_crop_size_ > 0) {
        crop_width = utils::NextUInt32(max_crop_size_ - min_crop_size_ + 1) + \
                                       min_crop_size_;
        crop_height = crop_width * (1 + utils::NextDouble() * \
                                                      max_aspect_ratio_ * 2 - max_aspect_ratio_);
        crop_height = std::max(min_crop_size_, std::min(crop_height, max_crop_size_));
      } else {
        crop_width = shape_[2];
        crop_height = shape_[1];
      }
      mshadow::index_t topleft_y;
      mshadow::index_t topleft_x;
      if (rand_crop_ != 0) {
        topleft_y = utils::NextUInt32(res.rows - crop_height + 1);
        topleft_x = utils::NextUInt32(res.cols - crop_width + 1);
      } else {
        topleft_y = cvRound((res.rows - crop_height) / 2.);
        topleft_x = cvRound((res.cols - crop_width) / 2.);
      }
      if (crop_y_start_ != -1) topleft_y = crop_y_start_;
      if (crop_x_start_ != -1) topleft_x = crop_x_start_;
      cv::Rect roi(topleft_x, topleft_y, crop_width, crop_height);
      res = res(roi);
      cv::resize(res, res, cv::Size(shape_[1], shape_[2]));
    }
    // Convert to mshadow tensor
    if (shape_[0] == 1) {
      for (index_t y = 0; y < shape_[1]; ++y) {
        for (index_t x = 0; x < shape_[2]; ++x) {
          img_[0][y][x] = res.at<uchar>(y, x);
        }
      }
    } else {
      for (index_t y = 0; y < shape_[1]; ++y) {
        for (index_t x = 0; x < shape_[2]; ++x) {
          // store in RGB order
          cv::Vec3b bgr = res.at<cv::Vec3b>(y, x);
          img_[0][y][x] = bgr[2];
          img_[1][y][x] = bgr[1];
          img_[2][y][x] = bgr[0];
        }
      }
    }
    // Subtract mean image
    if (!meanSubtracted && shape_ == meanimg_.shape_ && meanfile_ready_) {
      img_ -= meanimg_;
    }
    if (means_.size() > 0) {
      for (size_t i = 0; i < means_.size(); ++i)
        img_[i] -= means_[i];
    }
    // Mirror
    if ((rand_mirror_ != 0 && utils::NextDouble() < 0.5f) || mirror_ == 1) {
      img_ = mirror(img_);
    }
    // Scale
    img_ *= scale_;
    out_.data = img_;
    #else
    if (shape_[1] == 1) {
      img_ = d.data * scale_;
    } else {
      utils::Assert(d.data.size(1) >= shape_[1] && d.data.size(2) >= shape_[2],
                    "Data size must be bigger than the input size to net.");
      if (max_crop_size_ > 0 || min_crop_size_ > 0 || max_aspect_ratio_ > 0 || max_rotate_angle_ > 0 || max_shear_ratio_ > 0.0f || rotate_ > 0 || rotate_list_.size() > 0) {
        utils::Error("Unsupported data augmentation option");
      }
      // Determine cropping anchor point
      mshadow::index_t topleft_y;
      mshadow::index_t topleft_x;
      if (rand_crop_ != 0) {
        topleft_y = utils::NextUInt32(d.data.size(1) - shape_[1] + 1);
        topleft_x = utils::NextUInt32(d.data.size(2) - shape_[2] + 1);
      } else {
        topleft_y = cvRound((d.data.size(1) - shape_[1]) / 2.);
        topleft_x = cvRound((d.data.size(2) - shape_[2]) / 2.);
      }
      if (crop_y_start_ != -1) topleft_y = crop_y_start_;
      if (crop_x_start_ != -1) topleft_x = crop_x_start_;
      // Mean subtraction, cropping, mirroring and scaling
      if (meanfile_ready_) {
        // substract mean image
        if ((rand_mirror_ != 0 && utils::NextDouble() < 0.5f) || mirror_ == 1) {
          if (d.data.shape_ == meanimg_.shape_){
            img_ = mirror(crop(d.data - meanimg_, img_[0].shape_, topleft_y, topleft_x)) * scale_;
          } else {
            img_ = mirror(crop(d.data, img_[0].shape_, topleft_y, topleft_x) - meanimg_) * scale_;
          }
        } else {
          if (d.data.shape_ == meanimg_.shape_){
            img_ = crop(d.data - meanimg_, img_[0].shape_, topleft_y, topleft_x) * scale_ ;
          } else {
            img_ = (crop(d.data, img_[0].shape_, topleft_y, topleft_x) - meanimg_) * scale_;
          }
        }
      } else {
        if (means_.size() > 0) {
          for (size_t i = 0; i < means_.size(); ++i) {
            img_[i] -= means_[i];
          }
        }
        if (rand_mirror_ != 0 && utils::NextDouble() < 0.5f || mirror_ == 1) {
          img_ = mirror(crop(d.data, img_[0].shape_, topleft_y, topleft_x)) * scale_;
        } else {
          img_ = crop(d.data, img_[0].shape_, topleft_y, topleft_x) * scale_ ;
        }
      }
    }
    out_.data = img_;
    #endif
  }
  inline void CreateMeanImg(void) {
    if (silent_ == 0) {
      printf("cannot find %s: create mean image, this will take some time...\n", name_meanimg_.c_str());
    }
    time_t start = time(NULL);
    unsigned long elapsed = 0;
    size_t imcnt = 1;

    utils::Assert(this->Next(), "input iterator failed.");
    meanimg_.Resize(mshadow::Shape3(shape_[0], shape_[1], shape_[2]));
    mshadow::Copy(meanimg_, this->Value().data);
    while (this->Next()) {
      meanimg_ += this->Value().data; imcnt += 1;
      elapsed = (long)(time(NULL) - start);
      if (imcnt % 1000 == 0 && silent_ == 0) {
        printf("\r                                                               \r");
        printf("[%8lu] images processed, %ld sec elapsed", imcnt, elapsed);
        fflush(stdout);
      }
    }
    meanimg_ *= (1.0f / imcnt);
    utils::StdFile fo(name_meanimg_.c_str(), "wb");
    meanimg_.SaveBinary(fo);
    if (silent_ == 0) {
      printf("\nsave mean image to %s..\n", name_meanimg_.c_str());
    }
    this->BeforeFirst();
  }
private:
  /*! \brief base iterator */
  IIterator<DataInst> *base_;
  /*! \brief input shape */
  mshadow::Shape<3> shape_;
  /*! \brief output data */
  DataInst out_;
  /*! \brief skip read */
  int test_skipread_;
  /*! \brief silent */
  int silent_;
  /*! \brief scale of data */
  real_t scale_;
  /*! \brief whether we do random cropping */
  int rand_crop_;
  /*! \brief whether we do random mirroring */
  int rand_mirror_;
  /*! \brief whether we do nonrandom croping */
  int crop_y_start_;
  /*! \brief whether we do nonrandom croping */
  int crop_x_start_;
  /*! \brief number of overflow instances that readed in round_batch mode */
  int num_overflow_;
  /*! \brief mean image, if needed */
  mshadow::TensorContainer<cpu, 3> meanimg_;
  /*! \brief mean image in OpenCV format, if needed */
  #if CXXNET_USE_OPENCV
  cv::Mat meanOpenCVImage_;
  #endif
  /*! \brief temp space */
  mshadow::TensorContainer<cpu, 3> img_;
  /*! \brief mean image file, if specified, will generate mean image file, and substract by mean */
  std::string name_meanimg_;
  /*! \brief Indicate the max ratation angle for augmentation, we will random rotate */
  /*! \brief [-max_rotate_angle, max_rotate_angle] */
  int max_rotate_angle_;
  /*! \brief max aspect ratio */
  float max_aspect_ratio_;
  /*! \brief random shear the image [-max_shear_ratio, max_shear_ratio] */
  float max_shear_ratio_;
  /*! \brief max crop size */
  int max_crop_size_;
  /*! \brief min crop size */
  int min_crop_size_;
  /*! \brief mean values for all channels */
  std::vector<float> means_;
  /*! \brief whether mean file is ready */
  bool meanfile_ready_;
  /*! \brief whether to mirror the image */
  int mirror_;
  /*! \brief rotate angle */
  int rotate_;
  /*! \brief list of possible rotate angle */
  std::vector<int> rotate_list_;
  /*! \brief resizing selector */
  std::string resize_;
};  // class AugmentIterator
}  // namespace cxxnet
#endif
