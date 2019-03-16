library(arules)
library(arulesViz)
data(Groceries)
groceries <- Groceries


##Messing with the data and algorithm##
#### Getting the data ####
# #### Exploring the data ####
# summary(groceries)
# #shows the first 5 transactions
# inspect(groceries[1:5])
# #support level for the first three items in the groceries dataset
# itemFrequency(groceries[,1:3])
# #Shows that there are only 8 items in the dataset with 10% support
# itemFrequencyPlot(groceries, support=.1)
# #You can limit the number of items you want to see from the data using topN
# itemFrequencyPlot(groceries, topN=20)
# #sparse matrix for the first 5 transactions
# image(groceries[1:5])
# #Sparse matrix for a randmomly sampled set of transactions(random 100 transactions)
# image(sample(groceries, 100))
# #### Training a model on the data ####
# #set of zero rules 
# apriori(groceries)
# support_model <-apriori(groceries, parameter=list(supp=0.001,
#                                                   conf = 0.15))
# appearance = list(default="rhs",lhs="whole milk"),
# control = list(verbose=F))
#setting up some rules 


# The 5 rules you can find with the highest confidence
print("~~~~~~~~~~~~~~~~~~~~~~~")
confidence_model <- apriori(groceries, parameter = list(supp = 0.1, conf = .1))
#Show the top 5 rules, but only 2 digits
options(digits=2)
confidence <-sort(confidence_model, by="confidence", decreasing=TRUE)
inspect(confidence[1:5])
print("~~~~~~~~~~~~~~~~~~~~~~~")

# The 5 rules you can find with the highest support
print("~~~~~~~~~~~~~~~~~~~~~~~")
support_model <-apriori(groceries, parameter=list(supp=0.01, conf = 0.15))
support <- sort(support_model, by="support", decreasing=TRUE)
inspect(support[1:5])
print("~~~~~~~~~~~~~~~~~~~~~~~")

# The 5 rules you can find with the highest lift
print("~~~~~~~~~~~~~~~~~~~~~~~")
lift_model <-apriori(groceries, parameter=list(supp= 0.001, conf = 0.15))
lift <- sort(lift_model, by="lift", decreasing=TRUE)
inspect(lift[1:5])
print("~~~~~~~~~~~~~~~~~~~~~~~")

# The 5 rules you think are the most interesting
print("~~~~~~~~~~~~~~~~~~~~~~~")
interesting_model <-apriori(groceries, parameter=list(supp=0.001, conf = 0.15),
                            appearance = list(default="rhs",lhs="whole milk"),
                            control = list(verbose=F))
interesting <- sort(interesting_model, by="lift", decreasing=TRUE)
inspect(interesting[1:5])
print("~~~~~~~~~~~~~~~~~~~~~~~")

#### Evaluate Model parameters ####
#summary(grocery_rules)

#map out the rules 
#plot(rules_1_c,method="graph",interactive=TRUE,shading=NA)
#plot(rules_2_c,method="graph",interactive=TRUE,shading=NA)

