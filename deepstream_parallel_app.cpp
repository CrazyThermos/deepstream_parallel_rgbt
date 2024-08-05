/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>

#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gstnvdsinfer.h"
#include "deepstream_common.h"

#include "nvds_obj_encode.h"
#include "nvds_yml_parser.h"
#include "gst-nvmessage.h"
#include <yaml-cpp/yaml.h>

#define _PATH_MAX 1024
#define FILE_NAME_SIZE (1024)

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* By default, OSD process-mode is set to GPU_MODE. To change mode, set as:
 * 0: CPU mode
 * 1: GPU mode
 */
#define OSD_PROCESS_MODE 1

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 0

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }
GST_DEBUG_CATEGORY (NVDS_APP);


gchar pgie_classes_str[6][32] = {"people"
                                "car",
                                "bus",
                                "motorcycle",
                                "lamp",
                                "truck"
                                };

static gboolean PERF_MODE = FALSE;
static gboolean USE_POSTPROCESS = FALSE;
static gboolean USE_SGIE = FALSE;
static gboolean USE_VIDEOTEMP = FALSE;

gint frame_number = 0, frame_count = 0;

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
  } \
} while (0)

#define DEBUG_STAGE 0
/* tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
tiler_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  if (DEBUG_STAGE){
    g_print("tiler_src_pad_buffer_probe start");
  }
  GstBuffer *buf = (GstBuffer *) info->data;
  guint num_rects = 0; 
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvDsMetaList * l_frame = NULL;
  NvDsMetaList * l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

#if 1
  // Get original raw data
  GstMapInfo in_map_info;
  if (!gst_buffer_map (buf, &in_map_info, GST_MAP_READ)) {
      g_print ("Error: Failed to map gst buffer\n");
      gst_buffer_unmap (buf, &in_map_info);
      return GST_PAD_PROBE_OK;
  }
  NvBufSurface *surface = (NvBufSurface *)in_map_info.data;
#endif

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
    l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    if (DEBUG_STAGE){
      g_print("create surface\n");
    }
      //TODO for cuda device memory we need to use cudamemcpy
      NvBufSurfaceMap (surface, -1, -1, NVBUF_MAP_READ);
      /* Cache the mapped data for CPU access */
      NvBufSurfaceSyncForCpu (surface, 0, 0); //will do nothing for unified memory type on dGPU
      guint surface_height = surface->surfaceList[frame_meta->batch_id].height;
      guint surface_width = surface->surfaceList[frame_meta->batch_id].width;

      //Create Mat from NvMM memory, refer opencv API for how to create a Mat
      cv::Mat nv12_mat = cv::Mat(surface_height*3/2, surface_width, CV_8UC1, surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0],
      surface->surfaceList[frame_meta->batch_id].pitch);
      cv::Mat gray;
      // temp.copyTo(trans_mat);
      // cv::Mat dstROI = trans_mat(cv::Rect(0, 0, fusion_mat.cols, fusion_mat.rows));
      cv::cvtColor(nv12_mat, gray, cv::COLOR_YUV2GRAY_NV12);


      NvBufSurface *inter_buf = nullptr;
      NvBufSurfaceCreateParams create_params;
      create_params.gpuId  = surface->gpuId;
      create_params.width  = surface_width;
      create_params.height = surface_height;
      create_params.colorFormat = NVBUF_COLOR_FORMAT_GRAY8;
      create_params.layout = NVBUF_LAYOUT_PITCH;
    #ifdef __aarch64__
      create_params.memType = NVBUF_MEM_DEFAULT;
    #else
      create_params.memType = NVBUF_MEM_CUDA_UNIFIED;
    #endif
      //Create another scratch RGBA NvBufSurface
      if (NvBufSurfaceCreate (&inter_buf, 1,
        &create_params) != 0) {
        GST_ERROR ("Error: Could not allocate internal buffer ");
        return GST_PAD_PROBE_OK;
      }
      if(NvBufSurfaceMap (inter_buf, 0, -1, NVBUF_MAP_READ_WRITE) != 0)
        std::cout << "map error" << std::endl;
      NvBufSurfaceSyncForCpu (inter_buf, -1, -1);
      cv::Mat trans_mat = cv::Mat(surface_height, surface_width, CV_8UC1, inter_buf->surfaceList[frame_meta->batch_id].mappedAddr.addr[0],
    inter_buf->surfaceList[0].pitch);
  
      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
          l_obj = l_obj->next) {
          obj_meta = (NvDsObjectMeta *) (l_obj->data);
          if (obj_meta->class_id % 3 == 0) {
            obj_meta->rect_params.border_color.red = 1.0;
            obj_meta->rect_params.border_color.green = 0.0;
            obj_meta->rect_params.border_color.blue = 0.0;
            obj_meta->rect_params.border_color.alpha = 0.5;
          }
          if (obj_meta->class_id % 3 == 1) {
            obj_meta->rect_params.border_color.red = 0.0;
            obj_meta->rect_params.border_color.green = 1.0;
            obj_meta->rect_params.border_color.blue = 0.0;
            obj_meta->rect_params.border_color.alpha = 0.5;
          }
          if (obj_meta->class_id % 3 == 2) {
            obj_meta->rect_params.border_color.red = 0.0;
            obj_meta->rect_params.border_color.green = 0.0;
            obj_meta->rect_params.border_color.blue = 1.0;
            obj_meta->rect_params.border_color.alpha = 0.5;
          }
          if(obj_meta->rect_params.left > surface_width/2) continue;
          cv::Mat dstROI = gray(cv::Rect(obj_meta->rect_params.left, obj_meta->rect_params.top, 
          obj_meta->rect_params.width, obj_meta->rect_params.height));
          // cv::Mat weights = cv::Mat::ones(dstROI.size(), dstROI.type());
          cv::GaussianBlur(dstROI, dstROI, cv::Size(3, 3), 0);
          // cv::Mat edges;
          // cv::Canny(dstROI, edges, 100, 255, 3, false);
          // cv::addWeighted(dstROI, 1, weights, 30, 0, dstROI);

          // Prewitt
          const int kernel_size = 3;
          int xarr[kernel_size][kernel_size] = {{1,1,1},{0,0,0},{-1,-1,-1}};
          int yarr[kernel_size][kernel_size] = {{-1,0,1},{-1,0,1},{-1,0,1}};

          cv::Mat kernelx = cv::Mat(kernel_size, kernel_size, CV_8U, xarr);
          cv::Mat kernely = cv::Mat(kernel_size, kernel_size, CV_8U, yarr);
          cv::Mat x,y;
          cv::filter2D(dstROI, x, CV_8UC1, kernelx);
          cv::filter2D(dstROI, y, CV_8UC1, kernely);
          cv::convertScaleAbs(x, x);
          cv::convertScaleAbs(y, y);
          cv::addWeighted(x, 0.5, y, 0.5, 0, x);
          cv::normalize(x, x, 0, 255, cv::NORM_MINMAX);
          cv::threshold(x, x, 50, 255, cv::THRESH_TRUNC);
          cv::addWeighted(dstROI, 1, x, 1, 0, dstROI);
          num_rects++;
      }
      // gray.copyTo(trans_mat);
      if(!DEBUG_STAGE)
      g_print ("TILER: Frame Number = %d Number of objects = %d\n", frame_meta->frame_num, num_rects);

#if 0
      /* To verify  encoded metadata of cropped frames, we iterate through the
      * user metadata of each frame and if a metadata of the type
      * 'NVDS_CROP_IMAGE_META' is found then we write that to a file as
      * implemented below.
      */
      char fileFrameNameString[FILE_NAME_SIZE];
      const char *osd_string = "tiler";
      /* For Demonstration Purposes we are writing metadata to jpeg images of
        * the first 10 frames only.
        * The files generated have an 'OSD' prefix. */
      if (frame_number < 11) {
        NvDsUserMetaList *usrMetaList = frame_meta->frame_user_meta_list;
        FILE *file;
        int stream_num = 0;
        while (usrMetaList != NULL) {
          NvDsUserMeta *usrMetaData = (NvDsUserMeta *) usrMetaList->data;
          if (usrMetaData->base_meta.meta_type == NVDS_CROP_IMAGE_META) {
            snprintf (fileFrameNameString, FILE_NAME_SIZE, "%s_frame_%d_%d.jpg",
                osd_string, frame_number, stream_num++);
            NvDsObjEncOutParams *enc_jpeg_image =
                (NvDsObjEncOutParams *) usrMetaData->user_meta_data;
            /* Write to File */
            file = fopen (fileFrameNameString, "wb");
            fwrite (enc_jpeg_image->outBuffer, sizeof (uint8_t),
                enc_jpeg_image->outLen, file);
            fclose (file);
          }
          else if(usrMetaData->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META){
            NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) usrMetaData->user_meta_data;
            for (unsigned int i = 0; i < meta->num_output_layers; i++) {
              NvDsInferLayerInfo *info = &meta->output_layers_info[i];
              info->buffer = meta->out_buf_ptrs_host[i];
              if (meta->out_buf_ptrs_dev[i]) {
                cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                    info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
              }
            }
            size_t ch = meta->output_layers_info->inferDims.d[0];
            size_t height = meta->output_layers_info->inferDims.d[1];
            size_t width = meta->output_layers_info->inferDims.d[2];
            size_t o_count = meta->output_layers_info->inferDims.numElements;
            // cvcore::Image<cvcore::ImageType::RGB_F32> img(width, height, width * sizeof(float), (float *) meta->output_layers_info[0].buffer, TRUE);
            float *outputCoverageBuffer =(float *) meta->output_layers_info[0].buffer;
            uint8_t* uint8Buffer = (uint8_t *)malloc(o_count*sizeof(uint8_t));

            for(int o_index=0; o_index < o_count; o_index++){
              // outputCoverageBuffer[o_index] *= 255.0f;
              uint8Buffer[o_index] = static_cast<uint8_t>(std::min(std::max(outputCoverageBuffer[o_index] * 255.0f, 0.0f), 255.0f));
            }
            NvDsObjEncOutParams *enc_jpeg_image = (NvDsObjEncOutParams *)malloc(sizeof(NvDsObjEncOutParams));
            enc_jpeg_image->outBuffer = uint8Buffer;
            enc_jpeg_image->outLen = o_count;
            snprintf (fileFrameNameString, FILE_NAME_SIZE, "%s_frame_%d_%d.jpg",
                  osd_string, frame_number, stream_num++);
            file = fopen (fileFrameNameString, "wb");
            fwrite (enc_jpeg_image->outBuffer, sizeof (uint8_t),
                  enc_jpeg_image->outLen, file);
            fclose (file);
          // g_print ("SIZE: %ld \n", sizeof(*outputCoverageBuffer));
            // std::vector < NvDsInferLayerInfo >
            // outputLayersInfo (meta->output_layers_info,
            // meta->output_layers_info + meta->num_output_layers);
          }
          usrMetaList = usrMetaList->next;
        }
      }
#endif

#if 1
    if (DEBUG_STAGE){
      g_print("get output tensor\n");
    }
    NvDsUserMetaList *usrMetaList = frame_meta->frame_user_meta_list;
    if (usrMetaList != NULL) {
      NvDsUserMeta *usrMetaData = (NvDsUserMeta *) usrMetaList->data;

      if(usrMetaData->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META){
          
          NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) usrMetaData->user_meta_data;
          for (unsigned int i = 0; i < meta->num_output_layers; i++) {
            NvDsInferLayerInfo *info = &meta->output_layers_info[i];
            info->buffer = meta->out_buf_ptrs_host[i];
            if (meta->out_buf_ptrs_dev[i]) {
              cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                  info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
            }
          }

          //Create image from NVDSINFER_TENSOR_OUTPUT_META
          guint ch = meta->output_layers_info->inferDims.d[0];
          guint fusion_height = meta->output_layers_info->inferDims.d[1];
          guint fusion_width = meta->output_layers_info->inferDims.d[2];
          guint o_count = meta->output_layers_info->inferDims.numElements;
          guint onechannel_size = fusion_height * fusion_width;
          float *outputCoverageBuffer =(float *) meta->output_layers_info[0].buffer;
          cv::Mat fusion_mat; 
          using image_type = uint8_t;
          int image_format = CV_8UC1;
          image_type* uint8Buffer = (image_type *)malloc(o_count * sizeof(image_type));
          for(int o_index=0; o_index < o_count; o_index++){
            uint8Buffer[o_index] = static_cast<uint8_t>(std::min(std::max(outputCoverageBuffer[o_index] * 255.0f, 0.0f), 255.0f));
          }
          fusion_mat = cv::Mat(fusion_height, fusion_width, image_format, uint8Buffer, fusion_width);
          cv::resize(fusion_mat, fusion_mat,cv::Size(surface_width/2,surface_height/2));
          // char file_name[128];
          // sprintf(file_name, "resize_stream%2d_%03d_3.png", frame_meta->source_id, frame_number);
          // cv::imwrite(file_name, fusion_mat);
#if 0
          image_type* uint8Buffer_C1 = (image_type *)malloc(onechannel_size * sizeof(image_type));
          image_type* uint8Buffer_C2 = (image_type *)malloc(onechannel_size * sizeof(image_type));
          image_type* uint8Buffer_C3 = (image_type *)malloc(onechannel_size * sizeof(image_type));
          
          for(int o_index=0; o_index < onechannel_size; o_index++){
            uint8Buffer_C1[o_index] = uint8Buffer[o_index];
            uint8Buffer_C2[o_index] = uint8Buffer[o_index + onechannel_size];
            uint8Buffer_C3[o_index] = uint8Buffer[o_index + 2 * onechannel_size];

          }

          std::vector<cv::Mat> channels;
          for(int idx=2;idx>=0;idx--){
            cv::Mat dumpimg;
            if (idx == 0) dumpimg = cv::Mat(fusion_height, fusion_width, image_format, uint8Buffer_C1);
            else if (idx == 1) dumpimg = cv::Mat(fusion_height, fusion_width, image_format, uint8Buffer_C2);
            else dumpimg = cv::Mat(fusion_height, fusion_width, image_format, uint8Buffer_C3);
            channels.emplace_back(dumpimg);
          }
          cv::merge(channels, fusion_mat);
#endif

          
          // 将源矩阵复制到目标矩阵的ROI区域
          cv::Mat dstROI = gray(cv::Rect(0, fusion_height*2, fusion_width*2, fusion_height*2));

          fusion_mat.copyTo(dstROI);
          // cv::cvtColor(dstROInv12, temp, cv::COLOR_YUV2BGR_NV12);
          // char file_name[128];
          // sprintf(file_name, "fusion_stream%2d_%03d_3.png", frame_meta->source_id, frame_number);
          // cv::imwrite(file_name, gray);

        }
      }
    if (DEBUG_STAGE){
      g_print("NvBufSurfTransformRect\n");
    }
      gray.copyTo(trans_mat);
      NvBufSurfaceSyncForDevice(inter_buf, -1, -1);
      inter_buf->numFilled = 1;
      NvBufSurfTransformConfigParams transform_config_params;
      NvBufSurfTransformParams transform_params;
      NvBufSurfTransformRect src_rect;
      NvBufSurfTransformRect dst_rect;
      cudaStream_t cuda_stream;
      CHECK_CUDA_STATUS (cudaStreamCreate (&cuda_stream),
        "Could not create cuda stream");
      // transform_config_params.input_buf_count = 2;
      // transform_config_params.composite_flag = NVBUFSURF_TRANSFORM_COMPOSITE;

      transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
      transform_config_params.gpu_id = surface->gpuId;
      transform_config_params.cuda_stream = cuda_stream;
      /* Set the transform session parameters for the conversions executed in this
        * thread. */
      NvBufSurfTransform_Error err = NvBufSurfTransformSetSessionParams (&transform_config_params);
      if (err != NvBufSurfTransformError_Success) {
        std::cout <<"NvBufSurfTransformSetSessionParams failed with error "<< err << std::endl;
        return GST_PAD_PROBE_OK;
      }
      /* Set the transform ROIs for source and destination, only do the color format conversion*/
      src_rect = {0, 0, surface_width, surface_height };
      dst_rect = {0, 0, surface_width, surface_height};

      /* Set the transform parameters */
      transform_params.src_rect = &src_rect;
      transform_params.dst_rect = &dst_rect;
      transform_params.transform_flag =
        NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
          NVBUFSURF_TRANSFORM_CROP_DST;
      transform_params.transform_filter = NvBufSurfTransformInter_Default;

      /* Transformation format conversion, Transform rotated RGBA mat to NV12 memory in original input surface*/
      err = NvBufSurfTransform (inter_buf, surface, &transform_params);
      if (err != NvBufSurfTransformError_Success) {
        std::cout<< "NvBufSurfTransform failed with error %d while converting buffer" << err <<std::endl;
        return GST_PAD_PROBE_OK;
      }


    // nvds_copy_obj_meta();
    cudaStreamDestroy(cuda_stream);
    NvBufSurfaceUnMap(inter_buf, 0, 0);
    NvBufSurfaceDestroy(inter_buf);
    NvBufSurfaceUnMap(surface, 0, 0);
    gst_buffer_unmap(buf, &in_map_info);
#endif 

  }
  if (DEBUG_STAGE){
    g_print("tiler_src_pad_buffer_probe end\n");
  }
  frame_number++;
  return GST_PAD_PROBE_OK;
}


/* tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
pgie_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0; 
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;
    NvDsObjectMetaList* temp_ptr;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    // Get original raw data
    GstMapInfo in_map_info;
    if (!gst_buffer_map (buf, &in_map_info, GST_MAP_READ)) {
        g_print ("Error: Failed to map gst buffer\n");
        gst_buffer_unmap (buf, &in_map_info);
        return GST_PAD_PROBE_OK;
    }
    NvBufSurface *surface = (NvBufSurface *)in_map_info.data;

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        
              //TODO for cuda device memory we need to use cudamemcpy
        if(frame_meta->batch_id == 0){
          temp_ptr = frame_meta->obj_meta_list;
#if 0
          NvBufSurfaceMap (surface, -1, -1, NVBUF_MAP_READ);
          /* Cache the mapped data for CPU access */
          NvBufSurfaceSyncForCpu (surface, -1, -1); //will do nothing for unified memory type on dGPU
          guint surface_height = surface->surfaceList[frame_meta->batch_id].height;
          guint surface_width = surface->surfaceList[frame_meta->batch_id].width;

          //Create Mat from NvMM memory, refer opencv API for how to create a Mat
          cv::Mat nv12_mat = cv::Mat(surface_height*3/2, surface_width, CV_8UC1, surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0],
          surface->surfaceList[frame_meta->batch_id].pitch);

          cv::Mat gray;
          cv::cvtColor(nv12_mat, gray, cv::COLOR_YUV2GRAY_NV12);

          NvBufSurface *inter_buf = nullptr;
          NvBufSurfaceCreateParams create_params;
          create_params.gpuId  = surface->gpuId;
          create_params.width  = surface_width;
          create_params.height = surface_height;
          create_params.size = 0;
          create_params.colorFormat = NVBUF_COLOR_FORMAT_GRAY8;
          create_params.layout = NVBUF_LAYOUT_PITCH;
        #ifdef __aarch64__
          create_params.memType = NVBUF_MEM_DEFAULT;
        #else
          create_params.memType = NVBUF_MEM_CUDA_UNIFIED;
        #endif
          //Create another scratch RGBA NvBufSurface
          if (NvBufSurfaceCreate (&inter_buf, 1,
            &create_params) != 0) {
            GST_ERROR ("Error: Could not allocate internal buffer ");
            return GST_PAD_PROBE_OK;
          }
          if(NvBufSurfaceMap (inter_buf, 0, -1, NVBUF_MAP_READ_WRITE) != 0)
            std::cout << "map error" << std::endl;
          NvBufSurfaceSyncForCpu (inter_buf, -1, -1);
          cv::Mat inter_mat = cv::Mat(surface_height, surface_width, CV_8UC1, inter_buf->surfaceList[0].mappedAddr.addr[0],
            inter_buf->surfaceList[0].pitch);

          for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                  l_obj = l_obj->next) {
              obj_meta = (NvDsObjectMeta *) (l_obj->data);
              if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                  vehicle_count++;
                  num_rects++;
              }
              if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                  person_count++;
                  num_rects++;
              }

              cv::Mat dstROI = gray(cv::Rect(obj_meta->rect_params.left, obj_meta->rect_params.top, 
              obj_meta->rect_params.width, obj_meta->rect_params.height));
              // cv::Mat weights = cv::Mat::ones(dstROI.size(), dstROI.type());
              cv::GaussianBlur(dstROI, dstROI, cv::Size(3, 3), 0);
              // cv::Mat edges;
              // cv::Canny(dstROI, edges, 100, 255, 3, false);
              // cv::addWeighted(dstROI, 1, weights, 30, 0, dstROI);

              // Prewitt
              const int kernel_size = 3;
              int xarr[kernel_size][kernel_size] = {{1,1,1},{0,0,0},{-1,-1,-1}};
              int yarr[kernel_size][kernel_size] = {{-1,0,1},{-1,0,1},{-1,0,1}};

              cv::Mat kernelx = cv::Mat(kernel_size, kernel_size, CV_8U, xarr);
              cv::Mat kernely = cv::Mat(kernel_size, kernel_size, CV_8U, yarr);
              cv::Mat x,y;
              cv::filter2D(dstROI, x, CV_8UC1, kernelx);
              cv::filter2D(dstROI, y, CV_8UC1, kernely);
              cv::convertScaleAbs(x, x);
              cv::convertScaleAbs(y, y);
              cv::addWeighted(x, 0.5, y, 0.5, 0, x);
              cv::normalize(x, x, 0, 255, cv::NORM_MINMAX);
              cv::threshold(x, x, 50, 255, cv::THRESH_TRUNC);
              cv::addWeighted(dstROI, 1, x, 1, 0, dstROI);

          }

          gray.copyTo(inter_mat);

          NvBufSurfaceSyncForDevice(inter_buf, -1, -1);
          inter_buf->numFilled = 1;
          NvBufSurfTransformConfigParams transform_config_params;
          NvBufSurfTransformParams transform_params;
          NvBufSurfTransformRect src_rect;
          NvBufSurfTransformRect dst_rect;
          cudaStream_t cuda_stream;
          CHECK_CUDA_STATUS (cudaStreamCreate (&cuda_stream),
            "Could not create cuda stream");
          transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
          transform_config_params.gpu_id = surface->gpuId;
          transform_config_params.cuda_stream = cuda_stream;
          /* Set the transform session parameters for the conversions executed in this
            * thread. */
          NvBufSurfTransform_Error err = NvBufSurfTransformSetSessionParams (&transform_config_params);
          if (err != NvBufSurfTransformError_Success) {
            std::cout <<"NvBufSurfTransformSetSessionParams failed with error "<< err << std::endl;
            return GST_PAD_PROBE_OK;
          }
          /* Set the transform ROIs for source and destination, only do the color format conversion*/
          src_rect = {0, 0, surface_width, surface_height};
          dst_rect = {0, 0, surface_width, surface_height};

          /* Set the transform parameters */
          transform_params.src_rect = &src_rect;
          transform_params.dst_rect = &dst_rect;
          transform_params.transform_flag =
            NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
              NVBUFSURF_TRANSFORM_CROP_DST;
          transform_params.transform_filter = NvBufSurfTransformInter_Default;

          /* Transformation format conversion, Transform rotated RGBA mat to NV12 memory in original input surface*/
          err = NvBufSurfTransform (inter_buf, surface, &transform_params);
          if (err != NvBufSurfTransformError_Success) {
            std::cout<< "NvBufSurfTransform failed with error %d while converting buffer" << err <<std::endl;
            return GST_PAD_PROBE_OK;
          }

          // access the surface modified by opencv
          // cv::cvtColor(nv12_mat, rgba_mat, cv::COLOR_YUV2BGRA_NV12);
          //dump the original NvbufSurface
          NvBufSurfaceUnMap(inter_buf, 0, 0);
          NvBufSurfaceUnMap(surface, 0, 0);
          NvBufSurfaceDestroy(inter_buf);
          cudaStreamDestroy(cuda_stream);
#endif
        }
        else{
          if (temp_ptr!=NULL&& ((NvDsFrameMeta *)(l_frame->data))->obj_meta_list!=NULL)
          nvds_copy_obj_meta_list(temp_ptr, (NvDsFrameMeta *)(l_frame->data));

        }
      if(!DEBUG_STAGE)
        g_print ("PGIE: Frame Number = %d Number of objects = %d "
            "Vehicle Count = %d Person Count = %d\n",
            frame_meta->frame_num, num_rects, vehicle_count, person_count);
// #if 1
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = display_meta->text_params;
        txt_params->display_text = (char *)g_malloc0 (MAX_DISPLAY_LEN);
        // offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        // offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 2;
        txt_params->text_bg_clr.red = 1.0;
        txt_params->text_bg_clr.green = 1.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        NvOSD_RectParams *rect_params = display_meta->rect_params;
        rect_params->border_color.red = 1.0;
        rect_params->border_color.green = 0.0;
        rect_params->border_color.blue = 1.0;
        rect_params->border_color.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
// #endif

    }
    gst_buffer_unmap (buf, &in_map_info);	
    return GST_PAD_PROBE_OK;
}


static GstPadProbeReturn
sgie_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer ctx)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    // guint num_rects = 0; 
    NvDsObjectMeta *obj_meta = NULL;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    //NvDsDisplayMeta *display_meta = NULL;
    
    GstMapInfo inmap = GST_MAP_INFO_INIT;
    if (!gst_buffer_map (buf, &inmap, GST_MAP_READ)) {
    GST_ERROR ("input buffer mapinfo failed");
    return GST_PAD_PROBE_DROP;
    }
    NvBufSurface *ip_surf = (NvBufSurface *) inmap.data;
    gst_buffer_unmap (buf, &inmap);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
      for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list;
        l_user != NULL; l_user = l_user->next) {
        // if(frame_count <= 10) {
        //   NvDsObjEncUsrArgs frameData = { 0 };
        //   /* Preset */
        //   frameData.isFrame = 1;
        //   /* To be set by user */
        //   frameData.saveImg = TRUE;
        //   frameData.attachUsrMeta = TRUE;
        //   /* Set if Image scaling Required */
        //   frameData.scaleImg = FALSE;
        //   frameData.scaledWidth = 0;
        //   frameData.scaledHeight = 0;
        //   /* Quality */
        //   frameData.quality = 80;
        //   /* Main Function Call */
        //   nvds_obj_enc_process ((NvDsObjEncCtxHandle)ctx, &frameData, ip_surf, NULL, frame_meta);
        //   }
        }
        if (frame_count < 3){
          NvDsUserMetaList *usrMetaList = frame_meta->frame_user_meta_list;
          FILE *file;
          if(usrMetaList == NULL) continue;
          
          NvDsUserMeta *usrMetaData = (NvDsUserMeta *) usrMetaList->data;
          if(usrMetaData->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) continue;
          
          int stream_num = 0;
          char fileFrameNameString[FILE_NAME_SIZE];
          const char *osd_string = "sgie";
          // NvDsUserMetaList *usrMetaList = frame_meta->frame_user_meta_list;
          // NvDsUserMeta *usrMetaData = (NvDsUserMeta *)user_meta->user_meta_data;
        
          NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) usrMetaData->user_meta_data;
          for (unsigned int i = 0; i < meta->num_output_layers; i++) {
            NvDsInferLayerInfo *info = &meta->output_layers_info[i];
            info->buffer = meta->out_buf_ptrs_host[i];
            if (meta->out_buf_ptrs_dev[i]) {
              cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                  info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
            }
          }
          
          size_t ch = meta->output_layers_info->inferDims.d[0];
          size_t height = meta->output_layers_info->inferDims.d[1];
          size_t width = meta->output_layers_info->inferDims.d[2];
          size_t o_count = meta->output_layers_info->inferDims.numElements;
          // cvcore::Image<cvcore::ImageType::RGB_F32> img(width, height, width * sizeof(float), (float *) meta->output_layers_info[0].buffer, TRUE);
          float *outputCoverageBuffer =(float *) meta->output_layers_info[0].buffer;
          using image_type = uint8_t;
          int image_format = CV_8UC1;
          image_type* uint8Buffer = (image_type *)malloc(o_count * sizeof(image_type));
          image_type* uint8Buffer_C1 = (image_type *)malloc(height * width * sizeof(image_type));
          image_type* uint8Buffer_C2 = (image_type *)malloc(height * width * sizeof(image_type));
          image_type* uint8Buffer_C3 = (image_type *)malloc(height * width * sizeof(image_type));

          // std::ofstream outputFile("float_array.txt");
          // for (int i = 0; i < o_count; ++i) {
          // outputFile << outputCoverageBuffer[i] << std::endl;  // 每个数字后面添加换行符
          // }

          for(int o_index=0; o_index < o_count; o_index++){
            // outputCoverageBuffer[o_index] *= 255.0f;
            uint8Buffer[o_index] = static_cast<uint8_t>(std::min(std::max(outputCoverageBuffer[o_index] * 255.0f, 0.0f), 255.0f));
            // uint8Buffer[o_index] = outputCoverageBuffer[o_index] * 255.0f;
          }

          // for(int o_index=0; o_index < height * width; o_index++){
          //   uint8Buffer_C1[o_index] = uint8Buffer[o_index];
          //   uint8Buffer_C2[o_index] = uint8Buffer[o_index + height * width];
          //   uint8Buffer_C3[o_index] = uint8Buffer[o_index + 2 * height * width];

          // }

          // std::vector<cv::Mat> channels;
          // for(int idx=2;idx>=0;idx--){
          //   snprintf (fileFrameNameString, FILE_NAME_SIZE, "%s_frame_%d_%d_c%d.jpg",
          //       osd_string, frame_number, stream_num++, idx);
          //   cv::Mat dumpimg;
          //   if (idx == 0) dumpimg = cv::Mat(height, width, image_format, uint8Buffer_C1);
          //   else if (idx == 1) dumpimg = cv::Mat(height, width, image_format, uint8Buffer_C2);
          //   else dumpimg = cv::Mat(height, width, image_format, uint8Buffer_C3);
          //   channels.emplace_back(dumpimg);
          //   // cv::imwrite(fileFrameNameString, dumpimg);
          // }

          // cv::Mat rgbimg;
          // cv::merge(channels, rgbimg);
          // snprintf (fileFrameNameString, FILE_NAME_SIZE, "%s_frame_%d_%d_rgb.png",
          // osd_string, frame_number, stream_num++);
          // cv::imwrite(fileFrameNameString, rgbimg);

          // NvDsObjEncOutParams *enc_jpeg_image = (NvDsObjEncOutParams *)malloc(sizeof(NvDsObjEncOutParams));
          // enc_jpeg_image->outBuffer = uint8Buffer;
          // enc_jpeg_image->outLen = o_count;

          // file = fopen (fileFrameNameString, "wb");
          // fwrite (enc_jpeg_image->outBuffer, sizeof (uint8_t),
          //       enc_jpeg_image->outLen, file);
          // fclose (file);

        }

        g_print ("SGIE: Frame Number = %d\n", frame_meta->frame_num);
    }
    frame_count++;
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d\n", stream_id);
        }
      }
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  if (!caps) {
    caps = gst_pad_query_caps (decoder_src_pad, NULL);
  }
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  if (g_strrstr (name, "source") == name) {
        g_object_set(G_OBJECT(object),"drop-on-latency",true,NULL);
  }

}

static GstElement *
create_http_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *decode_bin = NULL, *souphttpsrc = NULL, *srctee=NULL, *srcqueue=NULL, *nvvidconv = NULL, * cap_filter = NULL, * cap_filter1 = NULL;
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  // if (PERF_MODE) {
  //   uri_decode_bin = gst_element_factory_make ("nvurisrcbin", "uri-decode-bin");
  //   g_object_set (G_OBJECT (uri_decode_bin), "file-loop", TRUE, NULL);
  //   g_object_set (G_OBJECT (uri_decode_bin), "cudadec-memtype", 0, NULL);
  // } else {
  //   uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");
  // }
  souphttpsrc = gst_element_factory_make("souphttpsrc","soup-http-src");
  if (!bin || !souphttpsrc) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }
  // g_signal_connect (G_OBJECT (souphttpsrc), "pad-added",
  //     G_CALLBACK (cb_newpad), bin);
  g_object_set (G_OBJECT (souphttpsrc), "location", uri, NULL);
  g_object_set (G_OBJECT (souphttpsrc), "is-live", true, NULL);
  g_object_set (G_OBJECT (souphttpsrc), "timeout", 5, NULL);
  g_object_set (G_OBJECT (souphttpsrc), "retry", 2, NULL);



  srctee = gst_element_factory_make ("tee", "srctee");
  if (!srctee) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", "srctee");
    return NULL;
  }

  srcqueue = gst_element_factory_make ("queue", "srcqueue");
  if (!srcqueue) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", "srcqueue");
    return NULL;
  }

  decode_bin = gst_element_factory_make ("decodebin", "decode-bin");
  if (!decode_bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", "decode-bin");
    return NULL;
  }

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);


  cap_filter = gst_element_factory_make ("queue", "cap_filter");
  if (!cap_filter) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", "cap_filter");
    return NULL;
  }


  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvidconv");
  if (!nvvidconv) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", "nvvidconv");
    return NULL;
  }
  caps = gst_caps_new_empty_simple("video/x-raw");
  feature = gst_caps_features_new("memory:NVMM", NULL);
  gst_caps_set_features(caps, 0, feature);


  cap_filter1 = gst_element_factory_make ("capsfilter", "cap_filter1_nvvidconv");
  if (!cap_filter1) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", "cap_filter1");
    return NULL;
  }
  g_object_set(G_OBJECT(cap_filter1), "caps", caps, NULL);
  gst_caps_unref(caps);

  gst_bin_add_many (GST_BIN (bin), souphttpsrc, srctee, srcqueue, decode_bin, cap_filter, nvvidconv, cap_filter1, NULL);
  gst_element_link_many(souphttpsrc, srctee, srcqueue, decode_bin, cap_filter, nvvidconv, cap_filter1, NULL);
  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL, *souphttpsrc=NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  if (PERF_MODE) {
    uri_decode_bin = gst_element_factory_make ("nvurisrcbin", "uri-decode-bin");
    g_object_set (G_OBJECT (uri_decode_bin), "file-loop", TRUE, NULL);
    g_object_set (G_OBJECT (uri_decode_bin), "cudadec-memtype", 0, NULL);
  } else {
    uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");
  }
  souphttpsrc = gst_element_factory_make("souphttpsrc","soup-http-src");
  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

/* Get the absolute path of a file mentioned in the config given a
 * file path absolute/relative to the config file. */
gboolean
get_absolute_file_path_yaml (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar abs_real_file_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  /* Absolute path. No need to resolve further. */
  if (file_path[0] == '/') {
    /* Check if the file exists, return error if not. */
    if (!realpath (file_path, abs_real_file_path)) {
      /* Ignore error if file does not exist and use the unresolved path. */
      if (errno != ENOENT)
        return FALSE;
    }
    g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
    return TRUE;
  }

  /* Get the absolute path of the config file. */
  if (!realpath (cfg_file_path, abs_cfg_path)) {
    return FALSE;
  }

  /* Remove the file name from the absolute path to get the directory of the
   * config file. */
  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  /* Get the absolute file path from the config file's directory path and
   * relative file path. */
  abs_file_path = g_strconcat (abs_cfg_path, file_path, nullptr);

  /* Resolve the path.*/
  if (realpath (abs_file_path, abs_real_file_path) == nullptr) {
    /* Ignore error if file does not exist and use the unresolved path. */
    if (errno == ENOENT)
      g_strlcpy (abs_real_file_path, abs_file_path, _PATH_MAX);
    else
      return FALSE;
  }

  g_free (abs_file_path);

  g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
  return TRUE;
}


char* get_plugin_config_path(const std::string name, const std::string key, gchar* cfg_file_path)
{
  gboolean find = FALSE;
  char *abs_path = (char*) malloc(sizeof(char) * 1024);
  // char *abs_path1 = (char*) malloc(sizeof(char) * 1024);

  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  for(YAML::const_iterator itr = configyml[name].begin();
     itr != configyml[name].end(); ++itr)
  {
    const std::string paramKey = itr->first.as<std::string>();
    if (paramKey == key) {
      find = TRUE;
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      strncpy(str, temp.c_str(), 1024);
      // config->config_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
            abs_path)) {
        g_printerr ("Error: Could not parse config-file-path in plugin.\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } 
      // std::cout <<  << << std::endl;

  }
  if(!find) {
    g_printerr("[WARNING] Can't found param in plugin:(%s)", key.c_str());
  }
done:
  if (!find) {
    g_printerr("failed");
    // std::cout <<  __func__ <<  << std::endl;
  }
  return abs_path;
}


gboolean
link_element_to_metamux_sink_pad (GstElement *metamux, GstElement *elem,
    gint index)
{
  gboolean ret = FALSE;
  GstPad *mux_sink_pad = NULL;
  GstPad *src_pad = NULL;
  gchar pad_name[16];

  if (index >= 0) {
    g_snprintf (pad_name, 16, "sink_%u", index);
    pad_name[15] = '\0';
  } else {
    strcpy (pad_name, "sink_%u");
  }

  mux_sink_pad = gst_element_get_request_pad (metamux, pad_name);
  if (!mux_sink_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get sink pad from metamux");
    goto done;
  }

  src_pad = gst_element_get_static_pad (elem, "src");
  if (!src_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get src pad from '%s'",
                        GST_ELEMENT_NAME (elem));
    goto done;
  }

  if (gst_pad_link (src_pad, mux_sink_pad) != GST_PAD_LINK_OK) {
    NVGSTDS_ERR_MSG_V ("Failed to link '%s' and '%s'", GST_ELEMENT_NAME (metamux),
        GST_ELEMENT_NAME (elem));
    goto done;
  }

  ret = TRUE;

done:
  if (mux_sink_pad) {
    gst_object_unref (mux_sink_pad);
  }
  if (src_pad) {
    gst_object_unref (src_pad);
  }
  return ret;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL,
             *streammux = NULL, 
             *streamdemux = NULL,
             *tee_b0=NULL,
             *tee_b1=NULL, 
             *tee0_queue_b0=NULL,
             *tee0_queue_b1=NULL,
             *tee1_queue_b0=NULL,
             *tee1_queue_b1=NULL,
             *streammux_b0 = NULL, 
             *streammux_b1 = NULL, 
             *sink = NULL, 
             *pgie_b0 = NULL, 
             *pgie_b1 = NULL, 
             *sgie = NULL, 
             *metamuxer=NULL,
             *postprocess=NULL, 
             *videotemplate=NULL,
             *nvvidconv = NULL,
             *nvosd = NULL, 
             *tiler = NULL, 
             *nvdslogger = NULL,
             *streamux_queue, 
             *pgie_queue_b0, 
             *pgie_queue_b1, 
             *sgie_queue,  
             *metamuxer_queue,
             *postprocess_queue, 
             *tiler_queue, 
             *nvvidconv_queue, 
             *videotemplate_queue, 
             *nvosd_queue;
              
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *tiler_src_pad = NULL, *pgie_src_pad = NULL, *sgie_src_pad = NULL;
  guint i =0, num_sources = 0;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size, sgie_batch_size;
  gboolean yaml_config = FALSE;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie_type = NVDS_GIE_PLUGIN_INFER;

  PERF_MODE = g_getenv("NVDS_TEST3_PERF_MODE") &&
      !g_strcmp0(g_getenv("NVDS_TEST3_PERF_MODE"), "1");
  USE_POSTPROCESS = g_getenv("NVDS_TEST3_USE_POSTPROCESS") &&
      !g_strcmp0(g_getenv("NVDS_TEST3_USE_POSTPROCESS"), "1");
  USE_SGIE = g_getenv("NVDS_TEST3_USE_SGIE") &&
      !g_strcmp0(g_getenv("NVDS_TEST3_USE_SGIE"), "1");
  USE_VIDEOTEMP = g_getenv("NVDS_TEST3_USE_VIDEOTEMP") &&
      !g_strcmp0(g_getenv("NVDS_TEST3_USE_VIDEOTEMP"), "1");
  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 2) {
    g_printerr ("Usage: %s <yml file>\n", argv[0]);
    g_printerr ("OR: %s <uri1> [uri2] ... [uriN] \n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Parse inference plugin type */
  yaml_config = (g_str_has_suffix (argv[1], ".yml") ||
          g_str_has_suffix (argv[1], ".yaml"));

  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1],
                "primary-gie"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&sgie_type, argv[1],
              "secondary-gie"));
  }

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest3-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  GList *src_list = NULL ;

  if (yaml_config) {

    RETURN_ON_PARSER_ERROR(nvds_parse_source_list(&src_list, argv[1], "source-list"));

    GList * temp = src_list;
    while(temp) {
      num_sources++;
      temp=temp->next;
    }
    g_list_free(temp);
  } else {
      num_sources = argc - 1;
  }

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };

    GstElement *source_bin= NULL;
    if (g_str_has_suffix (argv[1], ".yml") || g_str_has_suffix (argv[1], ".yaml")) {
      char * uri = (char*)(src_list)->data;
      g_print("Now playing : %s\n", uri);
      if(uri[0] =='h' && uri[1] =='t' && uri[2] =='t' && uri[3] =='p'){
           source_bin = create_http_bin (i, uri);
      }
      else source_bin = create_source_bin (i, uri);
    } else {
      source_bin = create_source_bin (i, argv[i + 1]);
    }
    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);

    if (yaml_config) {
      src_list = src_list->next;
    }
  }

  if (yaml_config) {
    g_list_free(src_list);
  }

  // preprocess0 = gst_element_factory_make ("nvdspreprocess", "preprocess0");
  // preprocess1 = gst_element_factory_make ("nvdspreprocess", "preprocess1");


  /* Use nvinfer or nvinferserver to infer on batched frame. */
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    pgie_b0 = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine-0");
    pgie_b1 = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine-1");

  } else {
    pgie_b0 = gst_element_factory_make ("nvinfer", "primary-nvinference-engine-0");
    pgie_b1 = gst_element_factory_make ("nvinfer", "primary-nvinference-engine-1");
  }


  streamdemux =  gst_element_factory_make ("nvstreamdemux", "nvdemux");
  g_object_set (G_OBJECT (streamdemux), "per-stream-eos", TRUE, NULL);

  tee_b0 = gst_element_factory_make ("tee", "tee_b0");
  tee_b1 = gst_element_factory_make ("tee", "tee_b1");

  streammux_b0 = gst_element_factory_make ("nvstreammux", "stream-muxer_b0");
  streammux_b1 = gst_element_factory_make ("nvstreammux", "stream-muxer_b1");

  tee0_queue_b0 = gst_element_factory_make ("queue", "tee0_queue_b0");
  tee0_queue_b1 = gst_element_factory_make ("queue", "tee0_queue_b1");
  tee1_queue_b0 = gst_element_factory_make ("queue", "tee1_queue_b0");
  tee1_queue_b1 = gst_element_factory_make ("queue", "tee1_queue_b1");

  metamuxer = gst_element_factory_make ("nvdsmetamux", "infer_bin_muxer");

  /* Add queue elements between every two elements */
  streamux_queue  = gst_element_factory_make ("queue", "streamux_queue");
  // preprocess_queue0  = gst_element_factory_make ("queue", "preprocess_queue0");
  // preprocess_queue1  = gst_element_factory_make ("queue", "preprocess_queue1");

  if (USE_POSTPROCESS){
    postprocess = gst_element_factory_make ("nvdspostprocess", "postprocess");
    postprocess_queue  = gst_element_factory_make ("queue", "postprocess_queue");
  }
  if (USE_VIDEOTEMP){
    videotemplate = gst_element_factory_make ("nvdsvideotemplate", "videotemplate");
    videotemplate_queue = gst_element_factory_make ("queue", "videotemplate_queue");
  }

  pgie_queue_b0      = gst_element_factory_make ("queue", "pgie_queue_0");
  pgie_queue_b1      = gst_element_factory_make ("queue", "pgie_queue_1");
  
  metamuxer_queue = gst_element_factory_make ("queue", "metamuxer_queue");
  sgie_queue      = gst_element_factory_make ("queue", "sgie_queue");
  tiler_queue     = gst_element_factory_make ("queue", "tiler_queue");
  nvvidconv_queue = gst_element_factory_make ("queue", "nvvidconv_queue");
  nvosd_queue     = gst_element_factory_make ("queue", "nvosd_queue");
  /* Use nvdslogger for perf measurement. */
  nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  if (PERF_MODE) {
    sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
  } else {
    /* Finally render the osd output */
    if(prop.integrated) {
      sink = gst_element_factory_make ("nv3dsink", "nv3d-sink");
    } else {
      sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
    }
  }

  if (!pgie_b0 || !pgie_b1 || (USE_SGIE&&!sgie) || (USE_POSTPROCESS&&!postprocess) || (USE_VIDEOTEMP&&!videotemplate) ||!nvdslogger || !tiler || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if (yaml_config) {

    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux, argv[1], "streammux"));
    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux_b0, argv[1],"streammux-b0"));
    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux_b1, argv[1],"streammux-b1"));

    // g_object_set (G_OBJECT (preprocess0), "config-file", preprocess_config_paths.first, NULL);
    if(USE_POSTPROCESS){
      char* postprocess_config_path = get_plugin_config_path("post-process", "postprocesslib-config-file", argv[1]);
      char* postprocess_lib_path = get_plugin_config_path("post-process", "postprocesslib-name", argv[1]);
      g_object_set (G_OBJECT (postprocess), "postprocesslib-config-file", postprocess_config_path, NULL);
      g_object_set (G_OBJECT (postprocess), "postprocesslib-name", postprocess_lib_path, NULL);
    }
    
    if(USE_VIDEOTEMP){
      // RETURN_ON_PARSER_ERROR(nvds_parse_postprocess(postprocess, argv[1], "post-process"));
      char* customlib_name = get_plugin_config_path("video-template", "customlib-name", argv[1]);
      // char* customlib_props = get_plugin_config("video-template", "customlib-props", argv[1]);
      g_object_set (G_OBJECT (videotemplate), "customlib-name", customlib_name, NULL);
      g_object_set (G_OBJECT (videotemplate), "customlib-props", "add-fusion-surface:0", NULL);

    }
  

    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie_b0, argv[1], "primary-gie-0"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie_b1, argv[1], "primary-gie-1"));

    if (USE_SGIE) RETURN_ON_PARSER_ERROR(nvds_parse_gie(sgie, argv[1], "secondary-gie"));


    g_object_get (G_OBJECT (pgie_b0), "batch-size", &pgie_batch_size, NULL);
    g_object_get (G_OBJECT (pgie_b1), "batch-size", &pgie_batch_size, NULL);

    if (pgie_batch_size != num_sources) {
      g_printerr
          ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
          pgie_batch_size, num_sources);
      g_object_set (G_OBJECT (pgie_b0), "batch-size", num_sources, NULL);
      g_object_set (G_OBJECT (pgie_b1), "batch-size", num_sources, NULL);
    }

    if (USE_SGIE) g_object_get (G_OBJECT (sgie), "batch-size", &sgie_batch_size, NULL);
    if (USE_SGIE && sgie_batch_size != num_sources) {
      g_printerr
          ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
          sgie_batch_size, num_sources);
      g_object_set (G_OBJECT (sgie), "batch-size", num_sources, NULL);
    }
    char * metamux_config_path = get_plugin_config_path("metamux","config-file-path",argv[1]);
    g_object_set (G_OBJECT (metamuxer), "config-file",GET_FILE_PATH (metamux_config_path), NULL);
    RETURN_ON_PARSER_ERROR(nvds_parse_osd(nvosd, argv[1], "osd"));

    tiler_rows = (guint) sqrt (num_sources);
    tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
    g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);

    RETURN_ON_PARSER_ERROR(nvds_parse_tiler(tiler, argv[1], "tiler"));
    if (PERF_MODE) {
        RETURN_ON_PARSER_ERROR(nvds_parse_fake_sink(sink, argv[1], "sink"));
    } else {
      if(prop.integrated) {
        RETURN_ON_PARSER_ERROR(nvds_parse_3d_sink(sink, argv[1], "sink"));
      } else {
        RETURN_ON_PARSER_ERROR(nvds_parse_egl_sink(sink, argv[1], "sink"));
      }
    }
  }
  else {

    g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
        MUXER_OUTPUT_HEIGHT,
        "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    /* Configure the nvinfer element using the nvinfer config file. */
    g_object_set (G_OBJECT (pgie_b0),
        "config-file-path", "dstest3_pgie_config.txt", NULL);
    g_object_set (G_OBJECT (pgie_b1),
        "config-file-path", "dstest3_pgie_config.txt", NULL);

    /* Override the batch-size set in the config file with the number of sources. */
    g_object_get (G_OBJECT (pgie_b0), "batch-size", &pgie_batch_size, NULL);
    g_object_get (G_OBJECT (pgie_b1), "batch-size", &pgie_batch_size, NULL);

    if (pgie_batch_size != num_sources) {
      g_printerr
          ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
          pgie_batch_size, num_sources);
      g_object_set (G_OBJECT (pgie_b0), "batch-size", num_sources, NULL);
      g_object_set (G_OBJECT (pgie_b1), "batch-size", num_sources, NULL);

    }

    tiler_rows = (guint) sqrt (num_sources);
    tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
    /* we set the tiler properties here */
    g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
        "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

    g_object_set (G_OBJECT (nvosd), "process-mode", OSD_PROCESS_MODE,
        "display-text", OSD_DISPLAY_TEXT, NULL);

    g_object_set (G_OBJECT (sink), "qos", 0, "type", 1 ,NULL);

  }

  if (PERF_MODE) {
      if(prop.integrated) {
          g_object_set (G_OBJECT (streammux), "nvbuf-memory-type", 4, NULL);
      } else {
          g_object_set (G_OBJECT (streammux), "nvbuf-memory-type", 2, NULL);
      }
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  // 
  // videotemplate, videotemplate_queue,
  gst_bin_add_many (GST_BIN (pipeline), streamux_queue, streamdemux, tee_b0, tee_b1, tee0_queue_b0, tee0_queue_b1, tee1_queue_b0, tee1_queue_b1, 
      streammux_b0, streammux_b1, pgie_b0, pgie_queue_b0, pgie_b1, pgie_queue_b1, metamuxer,  metamuxer_queue, nvdslogger, tiler,
      tiler_queue, nvvidconv, nvvidconv_queue, nvosd, nvosd_queue, sink, NULL);
  /* we link the elements together
  * source0 -> \ 
  * source1 -> -> nvinfer -> nvdslogger -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (USE_SGIE && USE_POSTPROCESS){
    gst_bin_add_many (GST_BIN (pipeline),sgie, sgie_queue, postprocess, postprocess_queue, NULL);
    if (!gst_element_link_many (streammux, streamux_queue, pgie_b0, pgie_queue_b0, sgie, sgie_queue, postprocess, postprocess_queue, 
                               nvvidconv, nvvidconv_queue, videotemplate, videotemplate_queue, nvdslogger, tiler, tiler_queue, 
                              nvosd, nvosd_queue, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
    }
  }
  else if(USE_SGIE){
    gst_bin_add_many (GST_BIN (pipeline),sgie, sgie_queue, NULL);
    if (!gst_element_link_many (streammux, streamux_queue, pgie_b0, pgie_queue_b0, sgie, sgie_queue,
                               nvvidconv, nvvidconv_queue, nvdslogger, tiler, tiler_queue, 
                              nvosd, nvosd_queue, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
    }
  }
  else if (USE_POSTPROCESS){
    gst_bin_add_many (GST_BIN (pipeline), postprocess, postprocess_queue, NULL);
    if (!gst_element_link_many (streammux, streamux_queue, pgie_b0, pgie_queue_b0, postprocess, postprocess_queue, 
                               nvvidconv, nvvidconv_queue, nvdslogger, tiler, tiler_queue, 
                              nvosd, nvosd_queue, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
    }
  }
  else if(USE_VIDEOTEMP){
    gst_bin_add_many (GST_BIN (pipeline), videotemplate, videotemplate_queue, NULL);
    if (!gst_element_link_many (streammux, streamux_queue, pgie_b0, pgie_queue_b0,  
                            nvvidconv, nvvidconv_queue, videotemplate, videotemplate_queue, nvdslogger, tiler, tiler_queue, 
                          nvosd, nvosd_queue, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
    }
  }
  else{
    if (!gst_element_link (streammux, streamux_queue))
    {
      g_printerr ("Elements queue1 to streamdemux could not be linked. Exiting.\n");
      return -1;
    }
    if (!gst_element_link (streamux_queue, streamdemux))
    {
      g_printerr ("Elements queue1 to streamdemux could not be linked. Exiting.\n");
      return -1;
    }
    /*----------------------------link streamdemux to tee and queue in different branch----------------------------------------------*/
    // branch0
    link_element_to_demux_src_pad(streamdemux, tee_b0, 0);
    link_element_to_tee_src_pad (tee_b0, tee0_queue_b0);
    link_element_to_tee_src_pad (tee_b0, tee0_queue_b1);

    //branch1
    link_element_to_demux_src_pad(streamdemux, tee_b1, 1);
    link_element_to_tee_src_pad (tee_b1, tee1_queue_b0);
    link_element_to_tee_src_pad (tee_b1, tee1_queue_b1);

    /*----------------------------link streamdemux to tee and queue in different branch----------------------------------------------*/

    /*----------------------------link  different queue to one streammux in different branch----------------------------------------------*/
    // branch0
    link_element_to_streammux_sink_pad (streammux_b0, tee0_queue_b0, 0);
    link_element_to_streammux_sink_pad (streammux_b0, tee1_queue_b0, 1);
    //branch1
    link_element_to_streammux_sink_pad (streammux_b1, tee0_queue_b1, 0);
    link_element_to_streammux_sink_pad (streammux_b1, tee1_queue_b1, 1);
    /*----------------------------link  different queue to one streammux in different branch----------------------------------------------*/

    /*----------------------------link streammux to pgie in different branch----------------------------------------------*/
    if (!gst_element_link (streammux_b0, pgie_b0))
    {
      g_printerr ("Elements streammux_b0 to pgie_b0 could not be linked. Exiting.\n");
      return -1;
    }
    if (!gst_element_link (streammux_b1, pgie_b1))
    {
      g_printerr ("Elements streammux_b0 to pgie_b0 could not be linked. Exiting.\n");
      return -1;
    }

    link_element_to_metamux_sink_pad (metamuxer, pgie_b0, 0);
    link_element_to_metamux_sink_pad (metamuxer, pgie_b1, 1);
    if (!gst_element_link (metamuxer, metamuxer_queue))
    {
      g_printerr ("Elements metamuxer to queue2 could not be linked. Exiting.\n");
      return -1;
    }


    if (!gst_element_link_many ( metamuxer_queue,
                            nvvidconv, nvvidconv_queue, nvdslogger, tiler, tiler_queue, 
                          nvosd, nvosd_queue, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
    }
  }
  // else{
  //   if (!gst_element_link_many (streammux, streamux_queue, pgie, pgie_queue,  
  //                       nvvidconv, nvvidconv_queue, nvdslogger, tiler, tiler_queue, 
  //                     nvosd, nvosd_queue, sink, NULL)) {
  //   g_printerr ("Elements could not be linked. Exiting.\n");
  //   return -1;
  //   }
  // }
  

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  
  tiler_src_pad = gst_element_get_static_pad (tiler, "src");
  if (!tiler_src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (tiler_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        tiler_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (tiler_src_pad);

  pgie_src_pad = gst_element_get_static_pad (pgie_b0, "src");
  // NvDsObjEncCtxHandle obj_ctx_handle = nvds_obj_enc_create_context (0);
  if (!pgie_src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        pgie_src_pad_buffer_probe, (gpointer) NULL, NULL);
  gst_object_unref (pgie_src_pad);

  // if(USE_SGIE){
  //   sgie_src_pad = gst_element_get_static_pad (sgie, "src");
  //   if (!sgie_src_pad)
  //     g_print ("Unable to get src pad\n");
  //   else
  //     gst_pad_add_probe (sgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
  //         sgie_src_pad_buffer_probe, NULL, NULL);
  //   gst_object_unref (sgie_src_pad);
  // }

  /* Set the pipeline to "playing" state */
  if (yaml_config) {
    g_print ("Using file: %s\n", argv[1]);
  }
  else {
    g_print ("Now playing:");
    for (i = 0; i < num_sources; i++) {
      g_print (" %s,", argv[i + 1]);
    }
    g_print ("\n");
  }
  gst_element_set_state (pipeline, GST_STATE_PLAYING);


  /* 管道可视化代码
    GstDebugGraphDetails:
    GST_DEBUG_GRAPH_SHOW_MEDIA_TYPE: show caps-name on edges
    GST_DEBUG_GRAPH_SHOW_CAPS_DETAILS: show caps-details on edges
    GST_DEBUG_GRAPH_SHOW_NON_DEFAULT_PARAMS: show modified parameters on elements
    GST_DEBUG_GRAPH_SHOW_STATES: show element states
    GST_DEBUG_GRAPH_SHOW_FULL_PARAMS: show full element parameter values even if they are very long
    GST_DEBUG_GRAPH_SHOW_ALL: show all the typical details that one might want
    GST_DEBUG_GRAPH_SHOW_VERBOSE: show all details regardless of how large or verbose they make the resulting output
  */
  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN (pipeline), GST_DEBUG_GRAPH_SHOW_MEDIA_TYPE, "pipeline-media-type");// 自加
  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN (pipeline), GST_DEBUG_GRAPH_SHOW_CAPS_DETAILS, "pipeline-caps-details");
  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN (pipeline), GST_DEBUG_GRAPH_SHOW_NON_DEFAULT_PARAMS, "pipeline-non-default-params");
  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN (pipeline), GST_DEBUG_GRAPH_SHOW_STATES, "pipeline-states");
  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN (pipeline), GST_DEBUG_GRAPH_SHOW_FULL_PARAMS, "pipeline-full-params");
  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN (pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline-all");
  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN (pipeline), GST_DEBUG_GRAPH_SHOW_VERBOSE, "pipeline-verbose");

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
