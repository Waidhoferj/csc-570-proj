from flask import Flask, jsonify, request

app= Flask(__name__)
@app.route("/recommend", methods=["POST"])
def get_reccos():
    if request.method=='POST':
        posted_data = request.get_json()
        data = posted_data['data']
        return jsonify(str("Successfully used  " + str(data)))

    
#  main thread of execution to start the server
if __name__=='__main__':
    app.run(debug=True)