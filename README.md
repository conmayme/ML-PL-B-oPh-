# ML-PL-B-oPh-
Đề tài này tập trung nghiên cứu và xây dựng các mô hình học máy nhằm phân loại mức
độ béo phì của từng cá nhân dựa trên các đặc trưng liên quan đến thói quen sinh hoạt
hằng ngày và các chỉ số thể chất.

Cụ thể, tôi tiến hành áp dụng một số thuật toán học máy tiêu biểu gồm K-Nearest Neighbors (KNN), Softmax Regression và Support Vector
Machine (SVM).

dataset : Tập dữ liệu gồm 2111 mẫu với 17 thuộc tính, bao gồm cả biến số, Biến nhị phân, Biến thứ bậc và biến phân loại. phù hợp cho các bài toán phân loại đa lớp, Biến đầu ra là nhãn phân loại đa lớp (7 lớp) biểu diễn mức độ béo phì.(Kaggle)

– LDA cho kết quả tốt nhất ở cả 3 mô hình, với accuracy dao động từ 0.9209 –
0.9274, F1-score cao nhất ở cả 3 mô hình, dao động 0.9211 – 0.9276,
khẳng định LDA là phương pháp biểu diễn dữ liệu tốt nhất cho bài toán này
– Dữ liệu gốc cho kết quả trung bình, accuracy khoảng 0.855 – 0.866, F1-score khoảng 0.8596 – 0.8659.
– PCA cho kết quả kém nhất, đặc biệt Softmax Regression chỉ đạt 0.6688, Softmax Regression chỉ đạt 0.6603,
cho thấy PCA làm giảm khả năng phân loại đáng kể.
