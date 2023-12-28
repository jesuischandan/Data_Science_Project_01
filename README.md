# Data_Science_Project_01
Here is the description of the project assigned by our Professor!

Help an international company Z to develop its innovative product.

Context: Company Z is an international company producing sanitary equipments. More precisely, the company designs, manufactures and distributes a wide range of products and systems for toilets: support frames, flush plates, flushing mechanisms, cisterns and seat covers.

The company employs over a thousand people across the world in 8 subsidiaries, 6 production facilities and 3 research and development centers.

The company aims to develop a new product designed for public places with the following core message: « A greater hygiene and comfort with the contactless flush solutions ».

The idea is to create a toilet for which the user does not have to touch the flush plate through an automatic flush system. This product is ideal for the hygiene and comfort of users.

Technical aspects: The photo below summarizes the innovative system developed by the company Z. The innovative system uses two LEDs that emitter (each) three light signals (red, blue, green) captured by two sensors that analyze the importance of light signals (see below)
![image](https://github.com/jesuischandan/Data_Science_Project_01/assets/103701334/cc9960df-e42c-4e2e-ac66-f644965fc5e6)

In order to automate flush volume according to what is in the toilet bowl, company’s engineers in R&D centers made numerous trials. Basically, in each trial they put different levels of waste (urine, paper, feces) in the toilet bowl, get the information send by sensors and finally determine the required flush volume to get a clean toilet. For each type of waste, level from 0 to 4 was used.
Data Science Task: You have access to trial datasets from the company’s Z laboratory and a brief description of the variables. In order to automate the contactless flush solution, the company needs a predictive model of the flush volume needed (Y – variable “Case of flush”) according to the information collected by the two sensors (X features/covariates – presented in green in the description file). This model will then be used and implemented in the final product.

Background: Company Z already developed (jointly with an external consultant) a predictive model (“competing model” hereafter). The company asks you to produce a new predictive model that significantly improves predictions. Your benchmark/baseline will thus be Company Z’s predictions. 

Your objective:  Develop the best predictive model with one important constraint: the model should produce homogeneous predictions over time (the function f that will link Y and X features should have a fixed set of parameters). 


Outputs to send me (no later than December 3, 24:00):

1.	A report explaining. 

1.1	The methodology, that is, the strategy you used to find the best model. You should include the results (table, graphs) you obtain at each step and interpret them.

1.2	The final predictions you obtain and an analysis of their quality.

1.3	An estimation of the environmental benefit (in terms of water saved and costs) of the product if it uses 1) your model or 2) the competing model. Use a concrete example like an Hotel, a company, a school, …

To achieve this, you will have to make some assumptions. To help you, you can use the following data: A classical toilet uses 6 liters of water per manual flush and the price of water is 4 euros per thousand liters. 

1.4	Discussion of potential improvements (next potential contract)

2.	Python code

3.	Presentation slides (summarize of your findings): Your oral presentation should not exceed 15-20 minutes.
