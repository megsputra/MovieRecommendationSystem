'''
PLEASE READ BEFORE RUNNING:

If you have any questions on how to run the algorithms separately, please contact
me at mputra@scu.edu.
'''

import numpy
import math
from collections import OrderedDict
import operator
from collections import defaultdict
from itertools import islice

#make 0 to nan

def predict_rating(infile,outfile, trainPy):
    print("Processing " + str(infile) + " please wait..")
    if infile == "test5.txt":
        n = 201
        k = 50
    elif infile == "test10.txt":
        n = 301
        k = 70
    elif infile == "test20.txt":
        n = 401
        k = 90
    
    t5py = numpy.zeros(shape = (100,1000))
    #t5py.fill(numpy.nan)
    with open(infile) as t5:
        t5Data = [i.strip().split(' ') for i in t5]  #[userId, movieId, rating]
        #t5Data = [[int(num) if num != 0 else float('nan') for num in sub] for sub in t5Data] #convert to int
        t5Data = [[int(num)  for num in sub] for sub in t5Data] #convert to int
    result = list()
    calcList = []
    active_user_movie = defaultdict(list)
    userRatingOnMovie = defaultdict(list)
    for data in t5Data:
        if data[2] == 0:
            calcList.append([data[0], data[1], int(0)])
            #t5py[data[0] - n][(data[1] - 1)] = float('nan')
        else:
            t5py[data[0] - n][(data[1] - 1)] = data[2]
            active_user_movie[data[0] - n].append(data[1]-1)
            userRatingOnMovie[data[0] - n].append(data[2])
    #print(calcList[0]) #[201, 1, 0]
   
   
    '''
    MAIN LOOP TO RATE THE 0's ON EACH USER
    '''
    for userToPredict in calcList:
        cur_user = userToPredict[0]- n
        target_movie = userToPredict[1] - 1
        neighbors = list()
        neighbor_similarity = {} #key = userid, value = similarity
        #find neighbor that has rated the targeted movie and all the movies that user has rated
        neighborDict = defaultdict(list)
        
        '''
        FINDING NEIGHBORS
        '''
        for user in range (0, len(trainPy)):
            if trainPy[user][target_movie] != 0: #if rated target movie 
                for movie in range(0, len(t5py[cur_user])): #for all the movies that user rated
                    if (trainPy[user][movie] != 0 and t5py[cur_user][movie] != 0): #if theres a common movie rated, add that user and the movie
                        neighborDict[user].append(movie)              
        # = userid : num common movies with cur user    
        #and then avg include the target movie
      
        
        '''
        FINDING SIMILARITY - COSINE SIMILARITY (IUF HERE)

        vn = []
        vu = []
        
        total_user = n + cur_user
        users_rated_target_movie = len(list(numpy.nonzero(trainPy[:, target_movie])))
        inverse_uf = iuf(total_user, users_rated_target_movie) 
        
        ##finding similarity using cosine similarity
        for neighbor, mov in neighborDict.items():
            for x in mov:
                vn.append(trainPy[neighbor][x])# * inverse_uf)
                vu.append(int(t5py[cur_user][x]))# * inverse_uf)
        
            cs = cosine_similarity(vn,vu)
            neighbor_similarity[neighbor] = cs
            vn.clear()
            vu.clear()
        #print("cur user:", cur_user, neighbor_similarity) 
        '''
        
        '''
        FINDING SIMILARITY - PEARSON COEFFICIENT
        
        r_a = {}
        samerating = 0
        user_vect = []
        neighbor_vect = []
        for nei, mov in neighborDict.items(): #for each neighbor
            for m in mov: #for each of the comommon movie
                user_vect.append(int(t5py[cur_user][m]))# * inverse_uf)
                neighbor_vect.append(int(trainPy[nei][m]))#* inverse_uf)
            #print(mag(neighbor_vect))
            #if mag(user_vect) == 0:  #ex user 206: 4, 4, 4, 4, 4
                ##use its avg as rating
                #break
            #if mag(neighbor_vect) > 0: 
            #neighbor_similarity[nei] = case_ampl(pearson(user_vect, neighbor_vect))
            #just call pearson if dont want case_ampl
            neighbor_similarity[nei] = pearson(user_vect, neighbor_vect)
            r_a[nei] = sum(neighbor_vect)/len(neighbor_vect)
            user_vect.clear()
            neighbor_vect.clear()     
        '''        

        a = t5py[cur_user][numpy.nonzero(t5py[cur_user])].mean()
        '''
        ITEM BASED
        '''
        #print(userRatingOnMovie)
        listofUsersRatedTarget = list(trainPy[:, target_movie])
        neighbor_similarity = {}
        #active_user_movie {#user: movies}
        for k, v in active_user_movie.items(): #for each movie user has rated
            #find users that have rated this movie and target movie
            for mo in v:          
                listofUsersRatedMo = list(trainPy[:, mo])
                nonZeroCM, nonZeroTarget = getNonZeroRatings(listofUsersRatedMo, listofUsersRatedTarget)
                if len(nonZeroCM) > 0:
                    neighbor_similarity[mo] = adj_cosine(nonZeroCM,nonZeroTarget, a)
                else : #no user that has ranked both movies
                    neighbor_similarity[mo] = 0


        '''
        SORTING THE SIMILARITIES


        neighbor_sorted = sorted(neighbor_similarity.items(), key=operator.itemgetter(1), reverse=True)
        if(len(neighbor_sorted) > k):
            k_neighbors = list(islice(neighbor_sorted, k))
        else:
            k_neighbors = neighbor_sorted
        knei = [nei for nei in k_neighbors if nei[1] > 0.001]
        #print(knei)
        '''
        '''
        DOING THE RATING PREDICTION USER BASED

        if len(knei) < 1:
            avge = a
        else:
            target_movie_rating = []
            weightforWA = []
            av_rating = []
            for m in knei:  
                target_movie_rating.append(int(trainPy[m[0]][target_movie]))
                weightforWA.append(m[1])
                #print(r_a[m[0]])
                av_rating.append(r_a[m[0]])       #uncomment out for pearson     
            avge = pcc(a, knei, target_movie_rating, av_rating) #comment out for non pearso
            avge2 = weighted_avg(weightforWA, target_movie_rating) 
            #print(avge)
        '''
        
        '''
        RATING PREDICTION FOR ITEM BASED
        '''

        avge = pcc_itembased(a, userRatingOnMovie[cur_user], active_user_movie[cur_user], neighbor_similarity)

        #p = (avge+avge2)/2
        p = int(round(avge))
        if p < 1: p = 1
        if p > 5: p = 5
        #print(p)
        result.append([userToPredict[0], target_movie + 1, p])

        
    '''
    WRITE RESULT TO OUTPUT FILE
    '''
    with open(outfile, 'w') as f:
        for r in result:
            for e in r:
                f.write('{} '.format(e))
            f.write("\n")    
    print("Done! Result writtent to", str(outfile))
    
    
'''
FUNCTION DEFINITIONS
'''
def getNonZeroRatings(l1,l2):
    m1 = list()
    m2 = list()
    
    for i in range(0,len(l1)):
        if l1[i] != 0 and l2[i] != 0:
            m1.append(int(l1[i]))
            m2.append(int(l2[i]))
    return m1,m2

def mag(x):
    s = 0
    avg = sum(x)/len(x)
    for i in x:
        s += (i-avg) ** 2
    return math.sqrt(s)
    #return math.sqrt(sum(i**2 for i in x))

def train():
    with open("train.txt") as trainfile:
        lst = [i.strip().split('\t') for i in trainfile]
        #trainList = [[int(num) if num != 0 else float('nan') for num in sub] for sub in lst] #convert to int
        trainList = [[int(num)  for num in sub] for sub in lst] #convert to int
    return numpy.array(trainList)
    
def cosine_similarity(v1, v2):
    num = numpy.dot(v1, v2)
    denom = sum_of_squares(v1) * sum_of_squares(v2)
    return num/denom
    
def sum_of_squares(vect):
    sum = 0
    for elem in vect:
        sum += (elem * elem)
    return math.sqrt(sum)    

def movie_rating_avg(kn, target):
    sum = 0
    norm = 0
    for e in kn: #rating * weight
        sum += trainPy[e[0]][target] * e[1]
    for i in kn:
        norm += i[1]
    return sum/norm
    
def weighted_avg(weight, ratings):
    num = 0
    den = 0
    for i in range(0, len(ratings)):
        num += ratings[i] * weight[i]
        den += weight[i]
    return num/den

def case_ampl(weight):
    p = 2.5
    return weight * abs(weight ** (p-1))

def iuf(totalU, totalUj):
    return math.log(totalU/totalUj)

def adj_cosine(targetrating, movrating, useravg):
    num = 0
    d1 = 0
    d2 = 0
    for i in range (0, len(targetrating)):
        num += (targetrating[i] - useravg) * (movrating[i] - useravg)
        d1 += (targetrating[i] - useravg) ** 2
        d2 += (movrating[i] - useravg) ** 2
    if( d1 == 0 or d2 == 0):
        return 0 # ignore user    
    return num / ( math.sqrt(d1) * math.sqrt(d2))
    
def pearson(user, neighbor):
    '''
    Calculates the pearson coefficient
    '''
    user_avg = sum(user)/len(user)
    neigh_avg = sum(neighbor)/len(neighbor)
    num = 0
    sq1 = 0
    sq2 = 0
    for i in range(0, len(user)):
        num += (user[i] - user_avg) * (neighbor[i] - neigh_avg)
        sq1 += (user[i] - user_avg) * (user[i] - user_avg)
        sq2 += (neighbor[i] - neigh_avg) * (neighbor[i] - neigh_avg)
    if( sq1 == 0 or sq2 == 0):
        return 0 # ignore user
    return abs((num / (math.sqrt(abs(sq1)) * math.sqrt(abs(sq2)))))

def pcc_itembased(userAvg, rac, mo1, sim):
    num = 0
    denom = 0
    for i in range(0, len(rac)):
        num += rac[i] * sim[mo1[i]]
        denom += sim[mo1[i]]
    if denom == 0:
        return userAvg
    return  (num/denom)
    #for k1 in kn :
        #avgratingofMovie = sum([pair[0] for pair in kn])
        #num = avgratingofMovie  * k1[1] #(weight)
    #denom = sum(abs(n) for _, n in kneighbor)
    #return useravg + (num/denom)

def pcc(user_avg, kneighbor, tr,ar):
    t = 0
    for i in range(0, len(tr)):
        t += (tr[i] - ar[i]) * kneighbor[i][1]
    avg_weight = sum(abs(n) for _, n in kneighbor)
    if (avg_weight == 0):
        print( "returning:", user_avg)
        return user_avg
    else:
        return user_avg + (t / avg_weight)

def main():
    print("Approx total processing time: 8 mins\nPlease be patient :)")
    predict_rating('test5.txt', 'result5.txt', trainPy)
    predict_rating('test10.txt', 'result10.txt', trainPy)
    predict_rating('test20.txt', 'result20.txt', trainPy)


trainPy = train()   
main()

'''
Pearson
MAE of GIVEN 5 : 0.892709766162311
MAE of GIVEN 10 : 0.7695
MAE of GIVEN 20 : 0.780746599787788
OVERALL MAE : 0.814726645870957

MAE of GIVEN 5 : 0.902963611354258
MAE of GIVEN 10 : 0.779333333333333
MAE of GIVEN 20 : 0.779106781132439
OVERALL MAE : 0.819816122147431

MAE of GIVEN 5 : 0.869701137926723
MAE of GIVEN 10 : 0.763333333333333
MAE of GIVEN 20 : 0.760875856081798
OVERALL MAE : 0.79720078804794
'''

'''
MAE of GIVEN 5 : 0.861197949230962
MAE of GIVEN 10 : 0.794833333333333
MAE of GIVEN 20 : 0.77003954856757
OVERALL MAE : 0.806066327368248
'''

#iuf is to compute weight
#iuf use original rating not rating mul by iuf
#iuf then ca

#for all common movies, mul by iuf -> similarity between users
'''
cosine + wa
MAE of GIVEN 5 : 0.839189696136051
MAE of GIVEN 10 : 0.793333333333333
MAE of GIVEN 20 : 0.768978489437639
OVERALL MAE : 0.798021671318339

MAE of GIVEN 5 : 0.822808553207453
MAE of GIVEN 10 : 0.7895
MAE of GIVEN 20 : 0.769460789042153
OVERALL MAE : 0.791906090953866
'''
'''
iuf + wa
MAE of GIVEN 5 : 0.861072902338377
MAE of GIVEN 10 : 0.795333333333333
MAE of GIVEN 20 : 0.769943088646667
OVERALL MAE : 0.806107371531768
'''
'''
MAE of GIVEN 5 : 0.822808553207453
MAE of GIVEN 10 : 0.7895
MAE of GIVEN 20 : 0.769460789042153
OVERALL MAE : 0.791906090953866
'''

'''
pearson + case ampli
MAE of GIVEN 5 : 0.888083031136676
MAE of GIVEN 10 : 0.792333333333333
MAE of GIVEN 20 : 0.796469566894955
OVERALL MAE : 0.825521260876703

Item based
MAE of GIVEN 5 : 0.869701137926723
MAE of GIVEN 10 : 0.763333333333333
MAE of GIVEN 20 : 0.760875856081798
OVERALL MAE : 0.79720078804794

'''