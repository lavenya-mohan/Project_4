from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Correct path to your Decision Tree model
model_path = 'decision_tree_model.pkl'

model = pickle.load(open(model_path,'rb'))

@app.route('/predict')
def predict():
    # Get data from the POST request
    data = [[-0.75179463, -0.56410297,  1.3097411 ,  0.63493804,  0.58982577,
        1.03239506,  1.15677133,  0.33016028,  0.28798923,  0.68344363,
       -0.28848722, -1.07628194,  0.33352921, -0.85142335,  0.35075302,
       -0.49210778, -0.17083614, -0.09894598, -0.10569424, -0.44356387,
       -0.41834871, -0.10431655, -1.85855918,  0.73460594,  0.43086418,
       -0.81454179, -0.91897115, -0.19997503, -0.72698865,  1.52226795,
        0.19688043,  0.19567519,  0.32108777, -0.66618076, -0.66618076,
        0.35633134, -0.01020301, -0.19973295, -0.18889542, -0.23432803,
       -0.35184367]]
    # Make prediction using the features provided in the request
    prediction = model.predict(np.array(data))
    # Return the prediction as a JSON
    x= prediction.tolist()
    if x[0]==0: 
        return 'No Future Claim'
    else: 
        return 'Future Claim successful'

if __name__ == '__main__':
    app.run(debug=True, port=5000)
