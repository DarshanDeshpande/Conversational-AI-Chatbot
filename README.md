# Conversational-AI-Chatbot
Transformer based Sequence model for contextual conversation generation <br>

<b>Sample Conversations:</b><br>
```
You: Hello there!
Bot:  Hi.
You: what would you do with a million dollars?
Bot:  I do not know.I just want to be with you.
You: what do you do for a living?
Bot:  I am a professional killer.
You: what did you just say?
Bot:  I am sorry , i did not mean to...
You: are you a man?
Bot:  I am not sure.I am just a messenger.
You: What is the longest that you've stayed awake for?
Bot:  I am not sure.I mean , i guess we should.
You: what is life?
Bot:  I do not know.I just do not know.
You: What is one thing you would like to become better at?
Bot:  I am not sure.I am just a little tired.I am too old for now.
You: How often do you nap?
Bot:  I am not sure i can face it.
You: If you were given three wishes, what would you wish for?
Bot:  I will take care of it.
```
<b>Warning: </b><br>
Conversations with the bot might lead to inappropriate or biased outcomes due to the nature of the training data (Cornell Corpus). User discretion is advised<br>

# How To Use
1. Switch to the Main Directory (Conversational-AI-Chatbot)
2. Install all requirements by executing <br>
```pip install -r requirements.txt``` <br>
3. Conversate with the bot by running the chatbot.py file through your preferred IDE or by typing the following command in your terminal <br>
```python chatbot.py``` <br>
 
# Training
Dataset: <a href="https://www.kaggle.com/Cornell-University/movie-dialog-corpus">Cornell Corpus</a> <br>
Samples: 197,158 Conversation Samples <br>
Max length of conversations: 40 words <br>
Vocab size: 14983 words <br>

<b>Hyperparameters:</b>
1. Learning Rate: Custom Scheduled <br>
2. Epochs: 225 <br>
3. Number of Encoder/Decoder layers: 3 <br>
4. Depth of Model: 256 <br>
5. Number of heads for Attention Model: 8 <br>

<b>Training Graph (Loss vs Epochs)</b><br><br>
<img src="https://github.com/DarshanDeshpande/Conversational-AI-Chatbot/blob/master/images/training-graph.png" title="Training Graph">

# Model Architecture
<img src="https://github.com/DarshanDeshpande/Conversational-AI-Chatbot/blob/master/images/Model.png" title="Transformer Model">

# Credits
1. Github User <a href="https://github.com/bryanlimy">bryanlimy</a> for his <a href="https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html">Blog</a> 
