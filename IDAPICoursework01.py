#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
    for row in theData:
      state = row[root]
      prior[state] += 1.0
    prior = map(lambda x: x/len(theData), prior)
    return array(prior)
  
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
    totals = zeros(noStates[varP], float)
    for row in theData:
      cPT[row[varC]][row[varP]] += 1.0
      totals[row[varP]] += 1.0
    x = []
    for row in cPT:
      x += [map(lambda x,y: x/y, row, totals)]
    return array(x)
  
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
    for row in theData:
      rowState = row[varRow]
      colState = row[varCol]
      jPT[rowState][colState] += 1.0
    total = len(theData)
    for i, row in enumerate(jPT):
      for j, col in enumerate(row):
        jPT[i][j] = col/total
    return array(jPT)
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    totals = zeros(len(aJPT[0]), float)
    j2c = []
    for row in aJPT:
      totals = map(lambda x,y: x+y, row, totals)
    for row in aJPT:
      j2c += [map(lambda x,y: x/y, row, totals)]
    return array(j2c)

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
    rootPdf = naiveBayes[0]

    naiveBayes = naiveBayes[1:]
    for i, tab in enumerate(naiveBayes):
        state = theQuery[i]
        for j, val in enumerate(prior):
            rootPdf[j] *= tab[state][j]
    total = sum(rootPdf)
    rootPdf = map(lambda x: x/total, rootPdf)

# Coursework 1 task 5 should be inserted here
  

# end of coursework 1 task 5
    return rootPdf
#
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
# Coursework 2 task 1 should be inserted here
   

# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    

# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    

# end of coursework 2 task 3
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
  
    return array(spanningTree)
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
   

# End of Coursework 3 task 1           
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList
# Coursework 3 task 2 begins here

# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here


# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here


# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here


# Coursework 3 task 5 ends here 
    return mdlAccuracy
#
# End of coursework 2
#
# Coursework 3 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 1
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
theData = array(datain)
prior = Prior(theData, 0, noStates)

cpt = CPT(theData, 2, 0, noStates)

jpt = JPT(theData, 2, 0, noStates)

j2cpt = JPT2CPT(jpt)

network = []
network += [Prior(theData, 0, noStates)]
for i in range(1, 6):
    network += [CPT(theData, i, 0, noStates)]

query = [4,0,0,0,5]
rpdf1 = Query(query, network)

network = []
network += [Prior(theData, 0, noStates)]
for i in range(1, 6):
    network += [CPT(theData, i, 0, noStates)]

query = [6,5,2,5,5]
rpdf2 = Query(query, network)


AppendString("IDAPIResults01.txt","Coursework One Results by af1410")
AppendString("IDAPIResults01.txt","") #blank line

AppendString("IDAPIResults01.txt","The prior probability of node 0")
AppendList("IDAPIResults01.txt", prior)
AppendString("IDAPIResults01.txt","") #blank line

AppendString("IDAPIResults01.txt","The conditional probability matrix P(2|0)")
AppendString("IDAPIResults01.txt", cpt)
AppendString("IDAPIResults01.txt","") #blank line

AppendString("IDAPIResults01.txt","The joint probability matrix P(2&0)")
AppendString("IDAPIResults01.txt", jpt)
AppendString("IDAPIResults01.txt","") #blank line

AppendString("IDAPIResults01.txt","The conditional probability matrix P(2|0) calculated from the joint probability matrix P(2&0)")
AppendString("IDAPIResults01.txt", j2cpt)
AppendString("IDAPIResults01.txt","") #blank line

AppendString("IDAPIResults01.txt","The result of Query [4,0,0,0,5]")
AppendList("IDAPIResults01.txt",array(rpdf1))
AppendString("IDAPIResults01.txt","") #blank line

AppendString("IDAPIResults01.txt","The result of Query [6,5,2,5,5]")
AppendList("IDAPIResults01.txt",array(rpdf2))