# dfsmn_causal_process
使用model scope dfsmn 模型： https://www.modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal 做的降噪，修改原始代码，采用流式输入，效果和原始网页一致



使用步骤：
1、pip install modelscope （使用原版的modelscope）
2、跑通原始dfsmn, my_process.py
3、修改modelscope中
modelscope.models.audio.ans.denoise_net import DfsmnAns
这一块的代码，将DfsmnAns改成DfsmnCausalProcess
需要修改denoise_net.py uni_deep_fsmn.py等文件,参考本git中上传的_modelscope文件
4、跑帧级推理： my_process_frame.py
