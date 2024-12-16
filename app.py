import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime
import pandas as pd
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import glob
import gc

# 设置页面配置
st.set_page_config(
    page_title="安全头盔检测系统",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session state
if 'all_results' not in st.session_state:
    st.session_state.all_results = []


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# 自定义CSS样式
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            background-color: #ff4b4b;
            color: white;
        }
        .stButton>button:hover {
            background-color: #ff6b6b;
        }
    </style>
""", unsafe_allow_html=True)


# 加载模型
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


# 模型选择
available_models = [f for f in os.listdir('.') if f.endswith('.pt')]
model_path = 'best.pt' if 'best.pt' in available_models else available_models[0] if available_models else None

if not model_path:
    st.error("未找到可用的模型文件")
    st.stop()

# 主界面
st.title("🪖 安全头盔检测系统")

# 侧边栏配置
with st.sidebar:
    st.title("⚙️ 系统配置")

    st.markdown("---")

    # 检测模式选择
    detection_mode = st.selectbox(
        "选择检测模式",
        ["📸 单张图片检测", "📁 批量图片检测", "📂 数据集文件夹检测", "🎥 实时视频检测", "📹 视频文件检测"],
        help="选择要使用的检测模式"
    )

    st.markdown("---")

    # 模型选择
    selected_models = st.multiselect(
        "选择要比较的模型",
        available_models,
        default=[model_path],
        help="可以选择多个模型进行比较"
    )

    if not selected_models:
        selected_models = [model_path]

    confidence = st.slider("置信度阈值", 0.0, 1.0, 0.01, 0.01)

    # 高级设置
    with st.expander("高级设置"):
        iou_threshold = st.slider("IOU阈值", 0.0, 1.0, 0.45, 0.01)
        max_det = st.number_input("最大检测数量", 1, 100, 20)

    st.markdown("---")

# 加载所有选中的模型
models = {name: load_model(name) for name in selected_models}


def show_model_comparison(results_data, model_names, processing_times, confidence_scores, detection_counts):
    """显示模型对比结果"""
    if len(model_names) > 1:
        st.markdown("### 模型性能比较")

        # 表格展示
        comparison_df = pd.DataFrame(results_data)
        st.table(comparison_df.style.highlight_max(subset=['检测目标数', '平均预测率'], color='lightgreen')
                 .highlight_min(subset=['处理时间'], color='lightgreen'))

        # 可视化比较
        st.markdown("### 可视化比较")

        # 使用两列布局来显示图表
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # 处理时间和预测率比较
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=model_names,
                y=processing_times,
                name="处理时间(秒)",
                yaxis="y1"
            ))
            fig.add_trace(go.Bar(
                x=model_names,
                y=[conf * 100 for conf in confidence_scores],
                name="平均预测率(%)",
                yaxis="y2"
            ))
            fig.update_layout(
                title="处理时间 vs 预测率",
                yaxis=dict(title="处理时间(秒)", side="left"),
                yaxis2=dict(title="预测率(%)", side="right", overlaying="y"),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        with viz_col2:
            # 雷达图比较
            max_time = max(processing_times) if processing_times else 1
            max_count = max(detection_counts) if detection_counts else 1
            
            # 如果max_count为0，将其设置为1以避免除以零
            if max_count == 0:
                max_count = 1

            fig_radar = go.Figure()
            for i, model in enumerate(model_names):
                fig_radar.add_trace(go.Scatterpolar(
                    r=[
                        confidence_scores[i] * 100,  # 预测率百分比
                        (1 - processing_times[i] / max_time) * 100,  # 处理速度（归一化并反转）
                        detection_counts[i] / max_count * 100  # 检测数量（归一化）
                    ],
                    theta=['预测率', '处理速度', '检测数量'],
                    name=model,
                    fill='toself'
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="模型综合性能对比"
            )
            st.plotly_chart(fig_radar, use_container_width=True)


def process_image(model, img_array, confidence, iou_threshold, max_det):
    """处理单张图片并返回结果"""
    start_time = time.time()
    
    # 确保图片是RGB格式
    if len(img_array.shape) == 2:  # 如果是灰度图
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # 如果是RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # 使用模型进行预测
    results = model.predict(
        source=img_array,
        conf=confidence,
        iou=iou_threshold,
        max_det=max_det,
        verbose=False
    )[0]
    
    # 获取处理时间
    process_time = time.time() - start_time
    
    # 获取检测结果
    boxes = results.boxes
    detection_count = len(boxes)
    avg_confidence = float(torch.mean(boxes.conf)) if detection_count > 0 else 0
    
    # 在图片上绘制检测框
    plotted_img = results.plot()
    
    return {
        'image': Image.fromarray(plotted_img),
        'process_time': process_time,
        'detection_count': detection_count,
        'avg_confidence': avg_confidence
    }

def process_dataset_batch(model, image_files, confidence, iou_threshold, max_det, batch_size=32):
    """分批处理数据集图片"""
    total_images = len(image_files)
    results_list = []
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total_images, batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_results = []
        
        for img_file in batch_files:
            try:
                # 读取并处理图片
                img = Image.open(img_file)
                img_array = np.array(img)
                result = process_image(model, img_array, confidence, iou_threshold, max_det)
                
                # 添加文件名到结果中
                result['filename'] = os.path.basename(img_file)
                batch_results.append(result)
                
                # 释放内存
                img.close()
                del img
                del img_array
                
            except Exception as e:
                st.warning(f"处理图片 {os.path.basename(img_file)} 时出错: {str(e)}")
                continue
        
        # 更新进度
        progress = (i + len(batch_files)) / total_images
        progress_bar.progress(progress)
        status_text.text(f"已处理 {i + len(batch_files)}/{total_images} 张图片")
        
        # 将批次结果添加到总结果列表
        results_list.extend(batch_results)
        
        # 强制垃圾回收
        gc.collect()
    
    progress_bar.empty()
    status_text.empty()
    
    return results_list

if detection_mode == "📸 单张图片检测":
    uploaded_file = st.file_uploader("上传图片", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        try:
            # 读取图片
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            # 创建多列布局显示不同模型的结果
            num_models = len(models)
            cols = st.columns(min(num_models, 2))
            
            results_data = []
            model_names = []
            processing_times = []
            confidence_scores = []
            detection_counts = []

            # 处理每个模型
            for idx, (model_name, model) in enumerate(models.items()):
                col_idx = idx % len(cols)
                
                with cols[col_idx]:
                    st.markdown(f"### 模型: {model_name}")
                    
                    # 处理图片
                    result = process_image(model, img_array, confidence, iou_threshold, max_det)
                    
                    # 显示处理后的图片
                    st.image(result['image'], caption=f"检测结果 - {model_name}")
                    
                    # 显示统计信息
                    st.write(f"处理时间: {result['process_time']:.3f} 秒")
                    st.write(f"检测目标数: {result['detection_count']}")
                    st.write(f"平均预测率: {result['avg_confidence']*100:.2f}%")
                    
                    # 收集比较数据
                    results_data.append({
                        '模型': model_name,
                        '处理时间': f"{result['process_time']:.3f}秒",
                        '检测目标数': result['detection_count'],
                        '平均预测率': f"{result['avg_confidence']*100:.2f}%"
                    })
                    model_names.append(model_name)
                    processing_times.append(result['process_time'])
                    confidence_scores.append(result['avg_confidence'])
                    detection_counts.append(result['detection_count'])
            
            # 显示模型比较结果
            show_model_comparison(results_data, model_names, processing_times, confidence_scores, detection_counts)

        except Exception as e:
            st.error(f"处理图片时出错: {str(e)}")

elif detection_mode == "📁 批量图片检测":
    # 初始化 session state
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    if 'last_params' not in st.session_state:
        st.session_state.last_params = None

    uploaded_files = st.file_uploader("选择多张图片", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    if uploaded_files:
        # 检查参数是否发生变化
        current_params = f"{confidence}_{iou_threshold}_{max_det}"
        params_changed = current_params != st.session_state.last_params
        
        # 如果是首次检测或参数发生变化，进行检测
        if not st.session_state.batch_results or params_changed:
            st.session_state.last_params = current_params
            st.info(f"已上传 {len(uploaded_files)} 张图片")

            # 开始批量检测
            all_results = []
            model_performance = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for img_idx, img_path in enumerate(uploaded_files):
                try:
                    # 更新进度
                    progress = (img_idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"处理图片 {img_idx+1}/{len(uploaded_files)}")

                    # 读取图片
                    image = Image.open(img_path)
                    img_array = np.array(image)

                    # 存储当前图片的所有模型结果
                    current_results = {
                        'image_name': img_path.name,
                        'results': []
                    }

                    # 使用每个选定的模型进行检测
                    for model_name, model in models.items():
                        start_time = time.time()
                        result = model(img_array, conf=confidence, iou=iou_threshold, max_det=max_det)
                        end_time = time.time()

                        # 计算性能指标
                        proc_time = end_time - start_time
                        num_detections = len(result[0].boxes)
                        if num_detections > 0:
                            confidences = [box.conf.cpu().numpy().item() for box in result[0].boxes]
                            avg_conf = sum(confidences) / num_detections
                        else:
                            avg_conf = 0

                        # 存储检测结果
                        current_results['results'].append({
                            'model': model_name,
                            'result': result,
                            'img': result[0].plot(),
                            'detections': num_detections,
                            'confidence': avg_conf,
                            'time': proc_time
                        })

                    all_results.append(current_results)

                    # 更新模型性能统计
                    if not model_performance:  # 首次初始化性能统计
                        model_performance = [{
                            '模型': model_result['model'],
                            '平均检测数': model_result['detections'],
                            '平均预测率': f"{model_result['confidence']:.2%}",
                            '平均处理时间': f"{model_result['time']:.3f}秒"
                        } for model_result in current_results['results']]
                    else:  # 更新现有统计
                        for i, model_result in enumerate(current_results['results']):
                            model_performance[i]['平均检测数'] = (model_performance[i]['平均检测数'] * len(
                                all_results) + model_result['detections']) / (len(all_results) + 1)
                            model_performance[i][
                                '平均预测率'] = f"{(float(model_performance[i]['平均预测率'].strip('%')) * len(all_results) + model_result['confidence'] * 100) / (len(all_results) + 1):.2f}%"
                            model_performance[i][
                                '平均处理时间'] = f"{(float(model_performance[i]['平均处理时间'].strip('秒')) * len(all_results) + model_result['time']) / (len(all_results) + 1):.3f}秒"

                except Exception as e:
                    st.error(f"处理图片 {img_path.name} 时出错: {str(e)}")

            # 清除进度条和状态文本
            progress_bar.empty()
            status_text.empty()

            # 保存结果到 session state
            st.session_state.batch_results = {
                'results': all_results,
                'performance': model_performance
            }
            st.session_state.current_image_index = 0

        # 显示检测结果
        results = st.session_state.batch_results['results']
        model_performance = st.session_state.batch_results['performance']

        # 创建上一张/下一张按钮
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("上一张"):
                st.session_state.current_image_index = (st.session_state.current_image_index - 1) % len(results)

        with col2:
            st.markdown(
                f"##### 当前: {st.session_state.current_image_index + 1}/{len(results)} - {results[st.session_state.current_image_index]['image_name']}")

        with col3:
            if st.button("下一张"):
                st.session_state.current_image_index = (st.session_state.current_image_index + 1) % len(results)

        # 显示当前图片的检测结果
        current_result = results[st.session_state.current_image_index]

        # 创建多列布局显示不同模型的结果
        num_models = len(current_result['results'])
        if num_models > 2:
            # 如果模型数量大于2，使用2列布局
            num_cols = 2
            num_rows = (num_models + 1) // 2
            for row in range(num_rows):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    model_idx = row * num_cols + col_idx
                    if model_idx < num_models:
                        model_result = current_result['results'][model_idx]
                        with cols[col_idx]:
                            st.markdown(f"**{model_result['model']}**")
                            st.image(model_result['img'])
        else:
            # 如果模型数量小于等于2，使用单行布局
            cols = st.columns(num_models)
            for idx, model_result in enumerate(current_result['results']):
                with cols[idx]:
                    st.markdown(f"**{model_result['model']}**")
                    st.image(model_result['img'])

        # 显示模型对比
        if len(model_performance) > 0:
            results_data = []
            model_names = []
            processing_times = []
            confidence_scores = []
            detection_counts = []
            for model_perf in model_performance:
                model_names.append(model_perf['模型'])
                detection_counts.append(model_perf['平均检测数'])
                conf = float(model_perf['平均预测率'].strip('%')) / 100
                confidence_scores.append(conf)
                time_val = float(model_perf['平均处理时间'].strip('秒'))
                processing_times.append(time_val)

                results_data.append({
                    "模型": model_perf['模型'],
                    "检测目标数": model_perf['平均检测数'],
                    "平均预测率": model_perf['平均预测率'],
                    "处理时间": model_perf['平均处理时间']
                })

            show_model_comparison(results_data, model_names, processing_times, confidence_scores, detection_counts)

elif detection_mode == "📂 数据集文件夹检测":
    dataset_path = st.text_input("输入数据集文件夹路径")
    
    if dataset_path and os.path.isdir(dataset_path):
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext.upper()}")))
        
        if not image_files:
            st.warning("未在指定文件夹中找到图片文件")
            st.stop()
        
        # 显示数据集信息
        st.info(f"找到 {len(image_files)} 张图片")
        
        # 批处理设置
        with st.expander("批处理设置"):
            batch_size = st.slider("批处理大小", 1, 64, 32, 
                                 help="较大的批处理大小可能会加快处理速度，但会消耗更多内存")
            
        # 创建结果保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(dataset_path, f"results_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # 处理每个模型
        all_model_results = {}
        
        for model_name, model in models.items():
            st.markdown(f"### 处理模型: {model_name}")
            
            try:
                # 分批处理数据集
                results = process_dataset_batch(
                    model, image_files, confidence, iou_threshold, max_det, batch_size
                )
                
                # 保存结果
                results_path = os.path.join(results_dir, f"{model_name}_results.csv")
                results_df = pd.DataFrame([{
                    '文件名': r['filename'],
                    '检测目标数': r['detection_count'],
                    '平均预测率': f"{r['avg_confidence']*100:.2f}%",
                    '处理时间(秒)': f"{r['process_time']:.3f}"
                } for r in results])
                
                results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
                
                # 显示统计信息
                st.write("统计信息:")
                st.write(f"- 总处理时间: {sum(r['process_time'] for r in results):.2f} 秒")
                st.write(f"- 平均每张处理时间: {sum(r['process_time'] for r in results)/len(results):.3f} 秒")
                st.write(f"- 总检测目标数: {sum(r['detection_count'] for r in results)}")
                st.write(f"- 结果已保存至: {results_path}")
                
                # 存储结果用于比较
                all_model_results[model_name] = results
                
            except Exception as e:
                st.error(f"处理模型 {model_name} 时出错: {str(e)}")
                continue
        
        # 如果有多个模型，显示比较结果
        if len(all_model_results) > 1:
            st.markdown("### 模型性能比较")
            
            # 准备比较数据
            comparison_data = []
            for model_name, results in all_model_results.items():
                total_time = sum(r['process_time'] for r in results)
                avg_time = total_time / len(results)
                total_detections = sum(r['detection_count'] for r in results)
                avg_confidence = sum(r['avg_confidence'] for r in results) / len(results)
                
                comparison_data.append({
                    '模型': model_name,
                    '总处理时间': f"{total_time:.2f}秒",
                    '平均处理时间': f"{avg_time:.3f}秒",
                    '总检测目标数': total_detections,
                    '平均预测率': f"{avg_confidence*100:.2f}%"
                })
            
            # 显示比较表格
            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df)
            
        st.success(f"所有结果已保存到: {results_dir}")
        
    else:
        st.warning("请输入有效的数据集文件夹路径")

elif detection_mode == "🎥 实时视频检测":
    st.markdown("### 实时视频检测")

    # 选择要对比的模型
    selected_models = st.multiselect(
        "选择要对比的模型",
        list(models.keys()),
        default=list(models.keys())[:2] if len(models) > 1 else list(models.keys())
    )

    if len(selected_models) < 1:
        st.warning("请至少选择一个模型")
    else:
        # 创建模型结果显示的列
        model_cols = st.columns(len(selected_models))
        model_displays = {model: col.empty() for model, col in zip(selected_models, model_cols)}
        model_stats = {model: col.empty() for model, col in zip(selected_models, model_cols)}

        # 初始化session state
        if 'detection_running' not in st.session_state:
            st.session_state.detection_running = False
        if 'detection_history' not in st.session_state:
            st.session_state.detection_history = []
        if 'current_session' not in st.session_state:
            st.session_state.current_session = {
                'start_time': None,
                'model_performance': {}
            }

        # 开始/停止按钮
        if not st.session_state.detection_running:
            if st.button("开始检测", key="start_detection"):
                st.session_state.detection_running = True
                # 记录会话开始时间
                st.session_state.current_session = {
                    'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'model_performance': {model_name: {
                        'detections': 0,
                        'confidence': 0,
                        'time': 0,
                        'frames': 0
                    } for model_name in selected_models}
                }
                st.rerun()
        else:
            if st.button("停止检测", key="stop_detection"):
                st.session_state.detection_running = False
                # 保存当前会话到历史记录
                if st.session_state.current_session['model_performance']:
                    session_data = {
                        'session_time': st.session_state.current_session['start_time'],
                        'model_stats': {}
                    }

                    for model_name, perf in st.session_state.current_session['model_performance'].items():
                        frames = perf['frames']
                        if frames > 0:
                            session_data['model_stats'][model_name] = {
                                "平均检测数": f"{perf['detections'] / frames:.1f}",
                                "平均预测率": f"{perf['confidence'] / frames:.2%}",
                                "平均帧率": f"{1.0 / (perf['time'] / frames):.1f} FPS"
                            }

                    st.session_state.detection_history.append(session_data)
                st.rerun()

        # 如果检测正在运行
        if st.session_state.detection_running:
            # 初始化摄像头
            cap = cv2.VideoCapture(0)

            try:
                if not cap.isOpened():
                    st.error("无法打开摄像头，请检查摄像头连接")
                    st.session_state.detection_running = False
                else:
                    st.success("摄像头已连接")

                    while cap.isOpened() and st.session_state.detection_running:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("无法读取摄像头画面")
                            break

                        for idx, model_name in enumerate(selected_models):
                            with model_cols[idx]:
                                model = models[model_name]
                                start_time = time.time()
                                result = model(frame, conf=confidence, iou=iou_threshold, max_det=max_det)
                                proc_time = time.time() - start_time

                                num_detections = len(result[0].boxes)
                                if num_detections > 0:
                                    confidences = [box.conf.item() for box in result[0].boxes]
                                    avg_conf = sum(confidences) / num_detections
                                else:
                                    avg_conf = 0

                                # 更新当前会话的性能统计
                                perf = st.session_state.current_session['model_performance'][model_name]
                                perf['detections'] += num_detections
                                perf['confidence'] += avg_conf
                                perf['time'] += proc_time
                                perf['frames'] += 1

                                # 显示检测结果
                                annotated_frame = result[0].plot()
                                model_displays[model_name].image(
                                    annotated_frame,
                                    channels="BGR",
                                    caption=f"{model_name} - 实时检测"
                                )

                                # 更新实时统计
                                current_fps = 1.0 / proc_time if proc_time > 0 else 0
                                stats_text = f"""
                                #### 实时统计 - {model_name}
                                - 检测目标数: {num_detections}
                                - 预测率: {avg_conf:.2%}
                                - 处理帧率: {current_fps:.1f} FPS
                                """
                                model_stats[model_name].markdown(stats_text)

                        time.sleep(0.01)

            finally:
                cap.release()

        # 显示历史检测记录
        if st.session_state.detection_history:
            st.markdown("### 历史检测记录")

            for idx, session in enumerate(reversed(st.session_state.detection_history)):
                st.markdown(f"#### 检测会话 {len(st.session_state.detection_history) - idx}")
                st.markdown(f"开始时间: {session['session_time']}")

                # 创建表格数据
                table_data = []
                for model_name, stats in session['model_stats'].items():
                    row_data = {"模型": model_name}
                    row_data.update(stats)
                    table_data.append(row_data)

                # 显示表格
                if table_data:
                    st.table(pd.DataFrame(table_data))
                st.markdown("---")

            # 添加清除历史记录的按钮
            if st.button("清除历史记录"):
                st.session_state.detection_history = []
                st.rerun()

elif detection_mode == "📹 视频文件检测":
    st.markdown("### 视频文件检测")

    # 视频上传
    video_file = st.file_uploader("上传视频", type=['mp4', 'avi', 'mov', 'mpeg'])

    if video_file is not None:
        # 创建临时文件来保存上传的视频
        temp_video_path = os.path.join(tempfile.gettempdir(), video_file.name)
        with open(temp_video_path, 'wb') as temp_file:
            temp_file.write(video_file.read())

        # 显示上传的视频信息
        video_info = f"视频名称: {video_file.name}"
        st.info(video_info)

        # 选择要对比的模型
        selected_models = st.multiselect(
            "选择要对比的模型",
            list(models.keys()),
            default=list(models.keys())[:2] if len(models) > 1 else list(models.keys())
        )

        if len(selected_models) < 1:
            st.warning("请至少选择一个模型")
        else:
            # 创建性能统计容器
            stats_container = st.empty()

            # 创建模型结果显示的列
            model_cols = st.columns(len(selected_models))
            model_displays = {model: col.empty() for model, col in zip(selected_models, model_cols)}
            model_stats = {model: col.empty() for model, col in zip(selected_models, model_cols)}

            if st.button("开始检测"):
                # 初始化视频捕获
                cap = cv2.VideoCapture(temp_video_path)

                # 获取视频信息
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                # 初始化性能统计
                model_performance = {model_name: {
                    'detections': 0,
                    'confidence': 0,
                    'time': 0,
                    'frames': 0
                } for model_name in selected_models}

                # 创建进度条
                progress_bar = st.progress(0)
                frame_counter = 0

                # 处理视频帧
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_counter += 1
                    progress = frame_counter / total_frames
                    progress_bar.progress(progress)

                    # 对每个选中的模型进行检测
                    for model_name in selected_models:
                        model = models[model_name]
                        start_time = time.time()
                        result = model(frame, conf=confidence, iou=iou_threshold, max_det=max_det)
                        proc_time = time.time() - start_time

                        # 计算性能指标
                        num_detections = len(result[0].boxes)
                        if num_detections > 0:
                            confidences = [box.conf.item() for box in result[0].boxes]
                            avg_conf = sum(confidences) / num_detections
                        else:
                            avg_conf = 0

                        # 更新模型性能统计
                        model_performance[model_name]['detections'] += num_detections
                        model_performance[model_name]['confidence'] += avg_conf
                        model_performance[model_name]['time'] += proc_time
                        model_performance[model_name]['frames'] += 1

                        # 在帧上绘制检测结果
                        annotated_frame = result[0].plot()

                        # 更新显示
                        model_displays[model_name].image(annotated_frame, channels="BGR", caption=model_name)

                        # 更新统计信息
                        current_fps = 1.0 / proc_time if proc_time > 0 else 0
                        stats_text = f"""
                        检测目标数: {num_detections}
                        平均预测率: {avg_conf:.2%}
                        处理帧率: {current_fps:.1f} FPS
                        """
                        model_stats[model_name].text(stats_text)

                    # 更新总体进度
                    stats_container.text(f"处理进度: {frame_counter}/{total_frames} 帧")

                # 释放资源
                cap.release()

                # 处理完成后显示总体性能统计和模型对比
                st.success("视频处理完成！")

                # 准备模型对比数据
                results_data = []
                model_names = []
                processing_times = []
                confidence_scores = []
                detection_counts = []

                # 计算和显示每个模型的平均性能
                st.markdown("### 模型性能统计")
                for model_name in selected_models:
                    perf = model_performance[model_name]
                    frames = perf['frames']
                    avg_detections = perf['detections'] / frames
                    avg_conf = perf['confidence'] / frames
                    avg_time = perf['time'] / frames
                    avg_fps = 1.0 / avg_time if avg_time > 0 else 0

                    model_names.append(model_name)
                    processing_times.append(avg_time)
                    confidence_scores.append(avg_conf)
                    detection_counts.append(avg_detections)

                    results_data.append({
                        "模型": model_name,
                        "平均检测数": f"{avg_detections:.1f}",
                        "平均预测率": f"{avg_conf:.2%}",
                        "平均帧率": f"{avg_fps:.1f} FPS"
                    })

                # 显示表格比较
                st.markdown("#### 性能数据对比")
                df = pd.DataFrame(results_data)
                st.table(df)

                # 显示图表对比
                st.markdown("#### 性能指标可视化")

                # 处理时间对比
                fig_time = go.Figure()
                for model_name, proc_time in zip(model_names, processing_times):
                    fig_time.add_trace(go.Bar(name=model_name, x=['处理时间'], y=[proc_time]))
                fig_time.update_layout(title="平均处理时间对比", yaxis_title="秒")
                st.plotly_chart(fig_time)

                # 预测率对比
                fig_conf = go.Figure()
                for model_name, conf in zip(model_names, confidence_scores):
                    fig_conf.add_trace(go.Bar(name=model_name, x=['预测率'], y=[conf]))
                fig_conf.update_layout(title="平均预测率对比", yaxis_title="预测率")
                st.plotly_chart(fig_conf)

                # 检测数量对比
                fig_det = go.Figure()
                for model_name, dets in zip(model_names, detection_counts):
                    fig_det.add_trace(go.Bar(name=model_name, x=['检测数'], y=[dets]))
                fig_det.update_layout(title="平均检测数对比", yaxis_title="数量")
                st.plotly_chart(fig_det)

                # 雷达图对比
                fig_radar = go.Figure()
                for model_name, proc_time, conf, dets in zip(model_names, processing_times, confidence_scores,
                                                             detection_counts):
                    # 归一化数据
                    max_time = max(processing_times)
                    max_dets = max(detection_counts)
                    normalized_time = 1 - (proc_time / max_time)  # 反转，使得更快的处理时间得分更高
                    normalized_dets = dets / max_dets

                    fig_radar.add_trace(go.Scatterpolar(
                        r=[normalized_time, conf, normalized_dets],
                        theta=['处理速度', '预测率', '检测数'],
                        name=model_name,
                        fill='toself'
                    ))

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="模型性能雷达图"
                )
                st.plotly_chart(fig_radar)

                # 清理临时文件
                try:
                    os.remove(temp_video_path)
                except:
                    pass
    else:
        st.info("请上传视频文件 (支持格式: MP4, AVI, MOV, MPEG)")

st.markdown("---")
st.markdown(" 第二小组 | " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
