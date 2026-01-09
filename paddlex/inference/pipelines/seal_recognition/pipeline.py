# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ....utils import logging
from ....utils.deps import pipeline_requires_extra
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ...models.object_detection.result import DetResult
from ...utils.benchmark import benchmark
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline
from ..components import CropByBoxes
from .result import SealRecognitionResult


@benchmark.time_methods
class _SealRecognitionPipeline(BasePipeline):
    """Seal Recognition Pipeline"""

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """Initializes the seal recognition pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use the high-performance
                inference plugin (HPIP) by default. Defaults to False.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                The default high-performance inference configuration dictionary.
                Defaults to None.
        """

        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_config=hpi_config
        )

        self.use_doc_preprocessor = config.get("use_doc_preprocessor", True)
        if self.use_doc_preprocessor:
            doc_preprocessor_config = config.get("SubPipelines", {}).get(
                "DocPreprocessor",
                {
                    "pipeline_config_error": "config error for doc_preprocessor_pipeline!"
                },
            )
            self.doc_preprocessor_pipeline = self.create_pipeline(
                doc_preprocessor_config
            )

        self.use_layout_detection = config.get("use_layout_detection", True)
        if self.use_layout_detection:
            layout_det_config = config.get("SubModules", {}).get(
                "LayoutDetection",
                {"model_config_error": "config error for layout_det_model!"},
            )
            layout_kwargs = {}
            if (threshold := layout_det_config.get("threshold", None)) is not None:
                layout_kwargs["threshold"] = threshold
            if (layout_nms := layout_det_config.get("layout_nms", None)) is not None:
                layout_kwargs["layout_nms"] = layout_nms
            if (
                layout_unclip_ratio := layout_det_config.get(
                    "layout_unclip_ratio", None
                )
            ) is not None:
                layout_kwargs["layout_unclip_ratio"] = layout_unclip_ratio
            if (
                layout_merge_bboxes_mode := layout_det_config.get(
                    "layout_merge_bboxes_mode", None
                )
            ) is not None:
                layout_kwargs["layout_merge_bboxes_mode"] = layout_merge_bboxes_mode
            self.layout_det_model = self.create_model(
                layout_det_config, **layout_kwargs
            )
        seal_ocr_config = config.get("SubPipelines", {}).get(
            "SealOCR", {"pipeline_config_error": "config error for seal_ocr_pipeline!"}
        )
        self.seal_ocr_pipeline = self.create_pipeline(seal_ocr_config)

        self._crop_by_boxes = CropByBoxes()

        self.batch_sampler = ImageBatchSampler(batch_size=config.get("batch_size", 1))

        self.img_reader = ReadImage(format="BGR")

    def check_model_settings_valid(
        self, model_settings: Dict, layout_det_res: DetResult
    ) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            model_settings (Dict): A dictionary containing input parameters.
            layout_det_res (DetResult): Layout detection result.

        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if model_settings["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized."
            )
            return False

        if model_settings["use_layout_detection"]:
            if layout_det_res is not None:
                logging.error(
                    "The layout detection model has already been initialized, please set use_layout_detection=False"
                )
                return False

            if not self.use_layout_detection:
                logging.error(
                    "Set use_layout_detection, but the models for layout detection are not initialized."
                )
                return False
        return True

    def get_model_settings(
        self,
        use_doc_orientation_classify: Optional[bool],
        use_doc_unwarping: Optional[bool],
        use_layout_detection: Optional[bool],
    ) -> dict:
        """
        Get the model settings based on the provided parameters or default values.

        Args:
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_layout_detection (Optional[bool]): Whether to use layout detection.

        Returns:
            dict: A dictionary containing the model settings.
        """
        if use_doc_orientation_classify is None and use_doc_unwarping is None:
            use_doc_preprocessor = self.use_doc_preprocessor
        else:
            if use_doc_orientation_classify is True or use_doc_unwarping is True:
                use_doc_preprocessor = True
            else:
                use_doc_preprocessor = False

        if use_layout_detection is None:
            use_layout_detection = self.use_layout_detection
        return dict(
            use_doc_preprocessor=use_doc_preprocessor,
            use_layout_detection=use_layout_detection,
        )

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_layout_detection: Optional[bool] = None,
        layout_det_res: Optional[Union[DetResult, List[DetResult]]] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        seal_det_limit_side_len: Optional[int] = None,
        seal_det_limit_type: Optional[str] = None,
        seal_det_thresh: Optional[float] = None,
        seal_det_box_thresh: Optional[float] = None,
        seal_det_unclip_ratio: Optional[float] = None,
        seal_rec_score_thresh: Optional[float] = None,
        **kwargs,
    ) -> SealRecognitionResult:

        model_settings = self.get_model_settings(
            use_doc_orientation_classify, use_doc_unwarping, use_layout_detection
        )
        if not self.check_model_settings_valid(model_settings, layout_det_res):
            yield {"error": "the input params for model settings are invalid!"}

        external_layout_det_results = layout_det_res
        doc_preprocessor_results=[]
        if external_layout_det_results is not None:
            if not isinstance(external_layout_det_results, list):
                external_layout_det_results = [external_layout_det_results]
            external_layout_det_results = iter(external_layout_det_results)

        for _, batch_data in enumerate(self.batch_sampler(input)):
            image_arrays = self.img_reader(batch_data.instances)

            # 1. 布局检测印章（在原始图像上）
            if model_settings["use_layout_detection"]:
                layout_det_results = list(
                    self.layout_det_model(
                        image_arrays,
                        threshold=layout_threshold,
                        layout_nms=layout_nms,
                        layout_unclip_ratio=layout_unclip_ratio,
                        layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                    )
                )
            elif external_layout_det_results is not None:
                layout_det_results = []
                for _ in image_arrays:
                    try:
                        layout_det_res = next(external_layout_det_results)
                    except StopIteration:
                        raise ValueError("No more layout det results")
                    layout_det_results.append(layout_det_res)
            else:
                layout_det_results = [{} for _ in image_arrays]

            if (
                not model_settings["use_layout_detection"]
                and external_layout_det_results is None
            ):
                # 没有布局检测时的处理
                if model_settings["use_doc_preprocessor"]:
                    doc_preprocessor_results = list(
                        self.doc_preprocessor_pipeline(
                            image_arrays,
                            use_doc_orientation_classify=use_doc_orientation_classify,
                            use_doc_unwarping=use_doc_unwarping,
                        )
                    )
                else:
                    doc_preprocessor_results = [{"output_img": arr} for arr in image_arrays]

                doc_preprocessor_images = [
                    item["output_img"] for item in doc_preprocessor_results
                ]
                
                layout_det_results = [{} for _ in doc_preprocessor_images]
                flat_seal_results = list(
                    self.seal_ocr_pipeline(
                        doc_preprocessor_images,
                        text_det_limit_side_len=seal_det_limit_side_len,
                        text_det_limit_type=seal_det_limit_type,
                        text_det_thresh=seal_det_thresh,
                        text_det_box_thresh=seal_det_box_thresh,
                        text_det_unclip_ratio=seal_det_unclip_ratio,
                        text_rec_score_thresh=seal_rec_score_thresh,
                    )
                )
                for seal_res in flat_seal_results:
                    seal_res["seal_region_id"] = 1
                seal_results = [[item] for item in flat_seal_results]
            else:
                # 2. 遍历所有印章区域
                seal_results = []
                for image_array, layout_det_res in zip(image_arrays, layout_det_results):
                    seal_results_for_img = []
                    seal_region_id = 1
                    
                    for box_info in layout_det_res["boxes"]:
                        if box_info["label"].lower() in ["seal"]:
                            # 3. 文档处理每一个印章
                            if model_settings["use_doc_preprocessor"]:
                                # 裁剪印章区域
                                crop_img_info = self._crop_by_boxes(image_array, [box_info])
                                crop_img = crop_img_info[0]["img"]
                                
                                # 对裁剪的印章区域进行文档预处理
                                doc_preprocessor_result = list(
                                    self.doc_preprocessor_pipeline(
                                        [crop_img],
                                        use_doc_orientation_classify=use_doc_orientation_classify,
                                        use_doc_unwarping=use_doc_unwarping,
                                    )
                                )
                                doc_preprocessor_results.append(doc_preprocessor_result[0])
                                processed_seal_img = doc_preprocessor_result[0]["output_img"]
                            else:
                                # 裁剪印章区域
                                crop_img_info = self._crop_by_boxes(image_array, [box_info])
                                processed_seal_img = crop_img_info[0]["img"]
                            
                            # 4. 印章识别
                            seal_ocr_result = list(
                                self.seal_ocr_pipeline(
                                    [processed_seal_img],
                                    text_det_limit_side_len=seal_det_limit_side_len,
                                    text_det_limit_type=seal_det_limit_type,
                                    text_det_thresh=seal_det_thresh,
                                    text_det_box_thresh=seal_det_box_thresh,
                                    text_det_unclip_ratio=seal_det_unclip_ratio,
                                    text_rec_score_thresh=seal_rec_score_thresh,
                                )
                            )
                            
                            if seal_ocr_result:
                                seal_res = seal_ocr_result[0]
                                seal_res["seal_region_id"] = seal_region_id
                                seal_results_for_img.append(seal_res)
                                seal_region_id += 1
                    
                    seal_results.append(seal_results_for_img)

            # 构建文档预处理结果用于返回（保持兼容性）
            # if model_settings["use_doc_preprocessor"]:
            #     doc_preprocessor_results = list(
            #         self.doc_preprocessor_pipeline(
            #             image_arrays,
            #             use_doc_orientation_classify=use_doc_orientation_classify,
            #             use_doc_unwarping=use_doc_unwarping,
            #         )
            #     )
            # else:
            #     doc_preprocessor_results = [{"output_img": arr} for arr in image_arrays]

            # 确保 seal_results 和 image_arrays 长度匹配
            if len(seal_results) < len(image_arrays):
                seal_results.extend([[] for _ in range(len(image_arrays) - len(seal_results))])

            for (
                input_path,
                page_index,
                doc_preprocessor_res,
                layout_det_res,
                seal_results_for_img,
            ) in zip(
                batch_data.input_paths,
                batch_data.page_indexes,
                doc_preprocessor_results,
                layout_det_results,
                seal_results,
            ):
                single_img_res = {
                    "input_path": input_path,
                    "page_index": page_index,
                    "doc_preprocessor_res": doc_preprocessor_res,
                    "layout_det_res": layout_det_res,
                    "seal_res_list": seal_results_for_img,
                    "model_settings": model_settings,
                }
                yield SealRecognitionResult(single_img_res)


@pipeline_requires_extra("ocr")
class SealRecognitionPipeline(AutoParallelImageSimpleInferencePipeline):
    entities = ["seal_recognition"]

    @property
    def _pipeline_cls(self):
        return _SealRecognitionPipeline

    def _get_batch_size(self, config):
        return config.get("batch_size", 1)
