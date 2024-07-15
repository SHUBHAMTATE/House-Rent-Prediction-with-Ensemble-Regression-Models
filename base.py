from flask import Flask,request,render_template
import pandas as pd
import pickle
#app
app=Flask(__name__)

def predict_input_data(input_df):
    # Load the column transformer and logistic regression model
    ct = pickle.load(open("coltransformer.pkl", "rb"))
    lr = pickle.load(open("logmodel.pkl", "rb"))
    # Apply the column transformer to the input data
    x = ct.fit_transform(input_df)
    print(x)


    # Make predictions
    ans = lr.predict(x)[0]

    return f"Your Rent Will be Approx   {round(ans,2)}₹ 🤓"


@app.route("/")
def display():
    return render_template("home.html")

@app.route("/predict",methods=["POST"])
def get_input_data():
    #Retrieve data from the form
    #['BHK', 'Size', 'Floor', 'City', 'Furnishing Status', 'Bathroom','Point of Contact']
    BHK=int(request.form.get("BHK"))
    Size=int(request.form.get("Size"))
    Floor=int(request.form.get("Floor"))
    City=request.form.get("City")
    Furnishing_Status=request.form.get("Furnishing Status")
    Bothroom=int(request.form.get("Bathroom"))
    Point_of_Contact=request.form.get("Point of Contact")
    # Create data frame with input data 

    input_df=pd.DataFrame(data=[[BHK,Size,Floor,City,Furnishing_Status,Bothroom,Point_of_Contact]],columns=['BHK', 'Size', 'Floor', 'City', 'Furnishing Status', 'Bathroom','Point of Contact'])
    
    ans=predict_input_data(input_df)
    return render_template("display.html",data=ans)

if __name__=="__main__":
    app.run(debug=True)
