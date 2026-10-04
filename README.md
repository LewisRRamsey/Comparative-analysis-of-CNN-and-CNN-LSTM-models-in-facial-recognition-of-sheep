# Comparative analysis of CNN and CNN-LSTM models in facial recognition of sheep (executive summary)

Facial recognition of sheep is a growing field of research that is being considered for widespread applications globally. Existing contact and non-contact methods for identifying livestock consistently present issues and leave room for vast improvement under new methods. Facial recognition models for sheep already exist in various forms (albeit the number of studies remain limited), but there is no comparison between CNN and CNN-LSTM models conducted by any individual.

The aim of this project is to investigate the performance and suitability of Convolutional Neural Networks (CNNs), and CNN – Long Short-Term Memory (CNN-LSTM) models in facial recognition of sheep. This will be done to draw a direct comparison between the two models for this task, which is yet to be considered. Furthermore, the project aims to investigate the effectiveness of the models in generalising on unseen test data when there is limited data available.

From this, the project aims to inform stakeholders of which is a more suitable machine learning model to select when considering this as an alternative approach to facial recognition of sheep, and if these models are suitable to be widely used by stakeholders for the task of identifying sheep through facial recognition.

The results from the project found that when evaluating the model under conditions with scarce data availability, the CNN-LSTM model performed better than the CNN in classifying individual sheep on unseen test data, when considering metrics such accuracy, precision and recall. The LSTM-CNN achieved a peak test accuracy of 74.66% compared to the CNN which achieved a peak test accuracy of 70.25%. Although it was found that both models were overfitting the data, but this still provided a base for comparison, as it was considered a limitation of the small dataset that caused this. In addition to this, the CNN-LSTM model had much larger overheads than the CNN model, with a training time that took over 3 times as long. 


<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/32859b24-c039-4df3-b4fd-1593c9147ca3" />


<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/4ec3a7ec-aec6-482e-a59b-fb93a2689968" />


To summarise, from the results and limitations of the project, it is not possible to answer definitively whether CNN-LSTM models are better than CNN models at facial recognition of sheep, but there is some evidence towards this question, successfully collected from the project by comparison of these two models, which answered that when using scarce data, CNN-LSTM models perform better at classifying unseen data than CNN models, even when not making use of temporal features. This suggests that LSTMs work potentially as a better spatial feature aggregator in the problem domain. Furthermore, it is not yet clear whether CNN-LSTM models are suitable to be deployed to stakeholders for facial recognition of sheep, due to their large overheads, limitations in the project, and further research needed; although indications of success other than this project come from previous studies in facial expression classification of humans, which is a similar domain.

