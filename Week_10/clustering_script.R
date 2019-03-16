library(datasets)
library(cluster)
library(dplyr)
library(tidyverse)
library(flextable)
library(knitr)
library(RColorBrewer)
library(animation)

##Step 1 
data = state.x77
data <- data.frame(data)


##Step 2
#Dendrogram of non-normalized data
# first compute a distance matrix
distance = dist(as.matrix(data))
# now perform the clustering
hc = hclust(distance)
# finally, plot the dendrogram
plot(hc)

##Step 3
#Dendrogram of normalized data
#summary(data)
#Scaling the data
data_scaled <- data %>%
  mutate(pop_scal = scale(Population),
         income_scal = scale(Income),
         illit_scal = scale(Illiteracy),
         life_scal = scale(Life.Exp),
         murder_scal = scale(Murder),
         grad_scal = scale(HS.Grad),
         frost_scal = scale(Frost),
         area_scal = scale(Area)) %>%
  select(-c(Population, Income, Illiteracy, Life.Exp, 
            Murder, HS.Grad, Frost, Area))

# distance matrix
distance_n = dist(as.matrix(data_scaled))
# perform clustering
hc_n = hclust(distance_n)
# plot the dendrogram
plot(hc_n)


##Step 4
data_scaled_a <- data_scaled %>% 
  select(-area_scal)
# compute a distance matrix
distance = dist(as.matrix(data_scaled_a))
# perform the clustering
hc_a = hclust(distance)
# plot the dendrogram
plot(hc_a)

##Step 5
data_scaled_f <- data_scaled[,7]
# compute a distance matrix
distance = dist(as.matrix(data_scaled_f))
# perform the clustering
hc_f = hclust(distance)
# plot the dendrogram
plot(hc_f)




###### K-means ######

##Step 1.
#use nomralized data 

##Step 2.
# Cluster into k=3 clusters:
pc_cluster = kmeans(data_scaled, 3)
summary(pc_cluster)
print(pc_cluster$cluster)
print(pc_cluster$centers)
print(pc_cluster$size)
clusplot(data_scaled, pc_cluster$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)


#Step 3.
# Cluster into k=3 clusters:
pc_cluster = kmeans(data_scaled, 3)
# compute the total within clusters sum of squares
kmean_withinss <- function(k) {
  cluster <- kmeans(data_scaled, k)
  return (cluster$tot.withinss)
}
# Set maximum cluster 
max_k <-25 
# Run algorithm over a range of k 
wss <- sapply(1:max_k, kmean_withinss)
elbow <-data.frame(1:max_k, wss)
ggplot(elbow, aes(x = X1.max_k, y = wss)) +
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(1, 26, by = 1)) +
  theme_bw()

#Step 4.
#Best looks like 7
pc_cluster_2 = kmeans(data_scaled, 7)


##Step 5
print(pc_cluster_2$cluster)

##Step 6
clusplot(data_scaled, pc_cluster_2$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)

##Step 7
#get centers
center <-pc_cluster_2$centers
# create dataset with the cluster number
cluster <- c(1:7)
center_df <- data.frame(cluster, center)
# Reshape the data
center_reshape <- gather(center_df, features, values, pop_scal: area_scal)
#Colors for plotting 
# Create the palette
hm.palette <-colorRampPalette(rev(brewer.pal(10, 'RdYlGn')),space='Lab')
#Visualize the clusters
#see what the clusters look like.
ggplot(data = center_reshape, aes(x = features, y = cluster, fill = values)) +
  scale_y_continuous(breaks = seq(1, 7, by = 1)) +
  geom_tile() +
  coord_equal() +
  scale_fill_gradientn(colours = hm.palette(90)) +
  theme_classic()


#This is what the clustering algorithm looks like for every step on the data
set.seed(2345)
# par(mfrow=c(2,1))
kmeans.ani(data_scaled, 3)
# dev.off()
