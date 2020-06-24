# Conversational-AI-Chatbot
Transformer based sequence model for contextual conversation generation
<b>Warning: </b><br>
Conversations with the bot might lead to inappropriate or biased outcomes due to the nature of the training data (Cornell Corpus). User descretion is advised<br>

# How To Use
1. Switch to the Main Directory (Conversational-AI-Chatbot)
2. Install all requirements by executing <br>
```pip install -r requirements.txt``` <br>
3. Conversate with the bot by running the chatbot.py file through your preferred IDE or by typing the following command in your terminal <br>
```python chatbot.py``` <br>
 
#Training
Dataset: <a href="https://www.kaggle.com/Cornell-University/movie-dialog-corpus">Cornell Corpus</a> <br>
Samples: 197,158 Conversation Samples <br>
Max length of conversations: 40 words <br>
Vocab size: 14983 words <br>

Hyperparameters:
1. Learning Rate: Custom Scheduled <br>
2. Epochs: 225 <br>
3. Number of Encoder/Decoder layers: 3 <br>
4. Depth of Model: 256 <br>
5. Number of heads for Attention Model: 8 <br>

# Model Architecture


# Credits
1. Github User <a href="https://github.com/bryanlimy">bryanlimy</a> for his <a href="https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html">Blog</a> 