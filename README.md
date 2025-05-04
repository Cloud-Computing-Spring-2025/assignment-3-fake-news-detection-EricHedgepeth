# Assignment-5-FakeNews-Detection


### what to install 

```bash 
pip install Faker
pip install pyspark
sudo apt-get update
sudo apt-get install -y openjdk-11-jdk
```

### Example data set after data generator 
```bash 
id,title,text,label
R129,New Report on Election,Board medical relate one. Still program week street space. Experts discuss the ongoing developments in election.,REAL
R227,New Report on Climate Change,Require foreign defense candidate dream type. Store call network system find million this. Activity carry ago agree phone fine choose. Experts discuss the ongoing developments in climate change.,REAL
F231,Aliens Confirmed by Sources,List will deep drive statement our support. Soon whose water never notice treat reflect senior. This just in — shocking claims about aliens surface on the internet.,FAKE
R066,New Report on Policy,Success citizen three itself information ever sit response. Seek into who open. Experts discuss the ongoing developments in policy.,REAL
F095,Mind Control Confirmed by Sources,Change same all perform image design. Half society year red change start another while. See you local participant glass. This just in — shocking claims about mind control surface on the internet.,FAKE
```


### tasks 

- ### Task 1: Load & Basic Exploration

- ### Task 2: Text Preprocessing

- ### Task 3: Feature Extraction

- ### Task 4: Model Training

- ### Task 5: Evaluate the Model


### How to run 

- ```bash 
spark-submit --master local[4] news_pipeline.py


or 


python3 news_pipeline.py

```