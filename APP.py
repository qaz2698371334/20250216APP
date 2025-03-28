# 导入需要的库
import streamlit as st
import pandas as pd
import joblib

# Streamlit 用户界面
st.markdown(
    "<h1 style='font-size: 16px; font-family: 宋体; text-align: center;'>基于机器学习的帕金森病检测系统</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h1 style='font-size: 16px; font-family: 宋体; '>请输入以下特征值：</h1>",
    unsafe_allow_html=True
)

a = st.number_input("PPE", step=0.01,value=0.15)
b = st.number_input("MDVP:Fo(Hz)",step=0.01,value=171.04)
c = st.number_input("MDVP:Fhi(Hz)", step=0.01,value=208.31)
d = st.number_input("spread2", step=0.01,value=0.23)
e = st.number_input("Jitter:DDP", step=0.01,value=0.01)
# 如果按下按钮
if st.button("预测"):  # 显示按钮
    # 加载训练好的模型
    model = joblib.load("XGBoost.pkl")
    # 将输入存储DataFrame
    X = pd.DataFrame([[a,b,c,d,e]],
                     columns = ['PPE', 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'spread2', 'Jitter:DDP'])
    # 进行预测
    prediction = model.predict(X)[0]
    Predict_proba = model.predict_proba(X)[:, 1][0]
    # 输出预测结果
    if prediction == 0:
        st.subheader(f"帕金森患者:  否")
    else:
        st.subheader(f"帕金森患者:  是")
    st.subheader(f"帕金森患者概率:  {'%.2f' % float(Predict_proba * 100) + '%'}")

