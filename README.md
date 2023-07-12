# CellMembraneSegmentation_MLOps

**Problem definition**

## Project contents
The contents of this project are available in the data and source folder. In the data folder, all the data extracted and used in this project are given. In the source folder, you can find the codes related to the demo project, data preprocessing, model implementation, and deployment-related codes. This project divided into 3 different phases: *Data Preprocessing* - *Model Development* - *Deployment*

### Phase1 - Data Preprocessing
One of the data sources used is the contents of [Persian Wikipedia](https://fa.wikipedia.org/wiki/%D8%B5%D9%81%D8%AD%D9%87%D9%94_%D8%A7%D8%B5%D9%84%DB%8C). Keywords appropriate to this field have been used to extract causal relationships. To extract possible sentences of cause and effect, sentences are first extracted from the existing texts and then the sentences containing the mentioned sentences are extracted for manual labeling.

Also, other source of data had used in this project, such as [PERLEX Dataset](http://farsbase.net/download/PERLEX.zip), [Causal-Bank Dataset](https://nlp.jhu.edu/causalbank/), [Causality corpus data of natural language processing laboratory of Shahid Beheshti University](http://nlp.sbu.ac.ir/), Data creation with ChatGPT.

Aggregation of labeled data for Bert, FastText, and NER models are found in samples.out.txt file. In this data, ✹Cause✹ - ✹✹Marker✹✹ - ✹✹✹Effect✹✹✹ used for initial labeling.

### Phase2 - Model development

#### Classification as preprocessing

#### Segmentation by U-net

### Phase3 - Deployment


### Demo
The demo of this project is written using the PyWebIO Python library. You can see a video of this demo in [Video_example of demo]()

## Usage
Deployment procedure

## Contributing
Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Team
[Amin Saeidi](https://github.com/Amin-Saeidi) | [Emad Deilam Salehi](https://github.com/Emad-Salehi)
## License
[MIT](https://choosealicense.com/licenses/mit/)
