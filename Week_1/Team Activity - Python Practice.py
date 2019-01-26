# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:10:57 2019

@author: kgund
"""
import random
import numpy as np

dictionary = {}

class Movie:
    
    def __init__(self, title="", year=0, runtime=0):    
            
        self.title = title
        self.year = year * (year >= 0)
        self.runtime = runtime * (runtime >= 0)   
                           
    def set_movie_data(self):
        self.title = str(input("Movie Title: "))
        self.year = int(input("Movie Year: "))
        self.runtime = int(input("Movie Runtime: "))
        self.year = self.year * (self.year >= 0)
        self.runtime = self.runtime * (self.runtime >= 0)
        
    def _repr_(self):
        rep = ("\nThe movie {} was made in {} and is {} minutes long" .format(self.title, 
              self.year, self.runtime))
        return rep
    
    def get_runtime(self):
        movie = str(self.title)
        hours = int(self.runtime / 60)
        minutes = int(self.runtime - hours * 60)
        hm = ("{} is {} hours and {} minutes\n" .format(movie, hours, minutes))
        return hm, movie, hours, minutes

def creat_movie_list():
    
    movie_list = [Movie("Jurassic World", 2015, 124), Movie("Harry Potter 1", 2014, 155),
                  Movie("Harry Potter2", 2013, 145), Movie("Harry Potter 3", 2012, 128),
                  Movie("Harry Potter 4", 2010, 175)]
    
    return movie_list 
      

    movie = Movie()
    #movie.set_movie_data()
    rep = movie._repr_()
    #print(rep)
    hm = movie.get_runtime()
    #print(hm)
    movie_list = creat_movie_list()

    filtered_list = []
    
    for m in movie_list:
        if m.runtime >= 150:
            filtered_list.append(m)
            
    for m in filtered_list:
        hm, movie, hours, minutes = m.get_runtime()
        print(m._repr_())
        print(hm)
        
        #stars_map = {m.title : random.uniform(0, 5) for m in movies}

    for m in movie_list:
        dictionary[m.title] = random.uniform(0, 5)
        
    for title in dictionary:
        print("{} - {:.2f} Stars".format(title, dictionary[title]))

    
def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100
        
        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array

def main():
    
    df = get_movie_data()
    print("{} rows {} columns".format(df.shape[0], df.shape[1]))

        
    print(df[0:2],"\n")
    
    print(df[:][-2:],"\n")
    
    
    print( df[:,2])
    


if __name__ == "__main__":
    main()

        

        