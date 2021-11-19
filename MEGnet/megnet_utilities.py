import os
import sys
import numpy as np
import pandas as pd
idx = pd.IndexSlice
from scipy.io import loadmat
import glob
sys.path.append('/archive/bioinformatics/DLLab/AlexTreacher/src')
import scipy.stats as stats
import paths
import json
import pickle

def fLoadData(strRatersLabels, strDBRoot, bCropSpatial=True, bAsSensorCap=False):
    #strRatersLabels = '/archive/bioinformatics/DLLab/AlexTreacher/data/MEGnet/AlexMcGillRatings.xlsx'
    #strDBRoot = '/archive/bioinformatics/DLLab/AlexTreacher/data/brainstorm_db/McGill'

    dfRatersLabels = pd.read_excel(strRatersLabels)
    intNICAComp = dfRatersLabels.columns[-1]

    lSpatial = []
    lTemporal = []
    lLabel = []
    lPath = []
    lSubject = []
    lScanType = []

    i = 0
    row = dfRatersLabels.iloc[i]
    for i, row in dfRatersLabels.iterrows():
        strDataPath = row['strPath']
        if bAsSensorCap:
            strDataPathSpatial = os.path.join(os.path.dirname(strDataPath), os.path.basename(strDataPath).replace('Disc','SensorCap'))
        else:
            strDataPathSpatial = strDataPath
        arrTemporalAll = loadmat(os.path.join(strDataPath, 'ICATimeSeries.mat'))['arrICATimeSeries'].T
        intComp = 1
        for intComp in range(1,intNICAComp+1):
            lLabel.append(row[intComp])
            lTemporal.append(arrTemporalAll[intComp-1]) #minus one because zero index, the 1st comp is at 0 index, the 2nd at 1 index etc.
            if bCropSpatial:
                lSpatial.append(loadmat(os.path.join(strDataPathSpatial,f'component{intComp}.mat'))['array'][30:-30,15:-14,:])
            else:
                lSpatial.append(loadmat(os.path.join(strDataPathSpatial,f'component{intComp}.mat'))['array'])
            lPath.append(row['strPath'])
            lSubject.append(row['strSubject'])
            lScanType.append(row['strType'])
        if i%20 == 0:
            print(f"Loading subject {i} of {dfRatersLabels.shape[0]}")
    return lSpatial, lTemporal, lLabel, lSubject, lPath, lScanType

def fLoadAllData(strDataRoot):
    pass

def fGetMeta(strSubject, strDatabase):
    #conversion for the gender provided
    dctSexConversion = {'M':'M',
                        'm':'M',
                        'Male':'M',
                        'male':'M',
                        'F':'F',
                        'f':'F',
                        'Female':'F',
                        'female':'F'}
    #get the meta data for all subjects from the 3 databases
    if strDatabase == 'HCP_MEG_20210212' or strDatabase == 'HCP':
        dfSubject = dctMetaFiles[strDatabase].loc[strSubject]
        return {'sex':dctSexConversion[dfSubject['Gender']],
                'age':dfSubject['Age']}
    elif strDatabase == 'iTAKL':
        dfDatabase = dctMetaFiles[strDatabase]
        lIndex = [x for x in dfDatabase.index.tolist() if strSubject in x]
        assert lIndex.__len__() <= 1
        try:
            dfSubject = dfDatabase.loc[lIndex[0]]
            #all of the iTAKL subjects are male
            return {'sex':'M',
                    'age':dfSubject['Age at Scan']}
        except IndexError:
            return {'sex':'M',
                    'age':np.nan}
    elif strDatabase == 'McGill':
        # a few subjects got renamed, so fix that issue with if statements (The file name on the MNI matches the meta)
        if strSubject == 'sub-0005':
            strSubject = 'sub-0001'
        try:
            dfSubject = dctMetaFiles[strDatabase].loc[[strSubject]].iloc[0]
            return {'sex':dctSexConversion[dfSubject['Gender']],
                   'age':dfSubject['Age at scan']}
        except KeyError: #if subject is not in meta
            return {'sex':np.nan,
                    'age':np.nan}
    else:
        raise ValueError(f'Database {strDatabase} not found')

def fBuildScanListsFromSubjectLists(dfSubjectList, dfCombinedMeta):
    """
    Use to build complete lists of scans given a lists of subjects in fSplitData()
    """
    dfScans = pd.DataFrame(columns = dfCombinedMeta.columns)
    for i,row in dfSubjectList.iterrows():
        strSubject = row['subject']
        if row['database'] == 'HCP_MEG_20210212':
            lSubjectIndex = dfCombinedMeta[dfCombinedMeta['subject']==row['subject']].index

        elif row['database'] == 'iTAKL':
            strSubject = '_'.join(row['subject'].split('_')[-2:])
            lSubjectIndex = dfCombinedMeta[dfCombinedMeta['subject'].apply(lambda x: strSubject in str(x))].index

        elif row['database'] == 'McGill':
            lSubjectIndex = dfCombinedMeta[dfCombinedMeta['subject']==row['subject']].index
        else:
            raise ValueError(f'Unknown database {row["database"]}')
        dfScans = dfScans.append(dfCombinedMeta.loc[lSubjectIndex])
    return dfScans

def fSplitStats(dfComplete, dfSplitMeta, lContinuousStratifiers = [], lCatagoricalStratifiers = []):
    dfStats = pd.Series()
    #split stats
    for strStrat in lContinuousStratifiers:
        lAge1 = dfComplete[strStrat]
        lAge2 = dfSplitMeta[strStrat]
        dT, dP = stats.ttest_ind(lAge1, lAge2, nan_policy='omit')
        dfStats[strStrat] = dP
    for strStrat in lCatagoricalStratifiers:
        lOptions = dfComplete[strStrat].unique().tolist()
        L1 = [dfComplete[dfComplete[strStrat] == x].shape[0] for x in lOptions]
        L2 = [dfSplitMeta[dfSplitMeta[strStrat] == x].shape[0] for x in lOptions]
        lContingencyTable = [L1,L2]
        stat, dP, dof, expected = stats.chi2_contingency(lContingencyTable)
        dfStats[strStrat] = dP
    return dfStats


def fSplitData(intK = 10,
               dTestFraction = .2,
               intRandomSeed = 55,
               lDatabases = ['HCP_MEG_20210212','iTAKL','McGill'],
               lPaths = [
                   os.path.join(paths.strDataDir, 'MEGnet/shared_ICA_raters/HCP_MEG_20210212'),
                   os.path.join(paths.strDataDir, 'MEGnet/shared_ICA_raters/iTAKL'),
                   os.path.join(paths.strDataDir, 'MEGnet/shared_ICA_raters/McGill')
                ],
               lRatings = [
                   os.path.join(paths.strDataDir, 'MEGnet/ratings/RG_LB_AP_HCP_MEG_20210212.xlsx'),
                   os.path.join(paths.strDataDir, 'MEGnet/ratings/RG_LB_AP_iTAKL.xlsx'),
                   os.path.join(paths.strDataDir, 'MEGnet/ratings/RG_LB_AP_McGill.xlsx')
               ],
               dctMetaFiles = {
                   'HCP_MEG_20210212':pd.read_excel(os.path.join(paths.strDataDir, 'MEGnet/HCP_MEG_Meta_AT.xlsx'), index_col=0),
                   'iTAKL':pd.read_excel(os.path.join(paths.strDataDir, 'MEGnet/iTAKL_BRP_age_at_scan.xlsx'), index_col=0),
                   'McGill':pd.read_csv(os.path.join(paths.strDataDir, 'MEGnet/McGillMeta_AT.csv'), index_col=1)
               },
               strOutDir = os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit')
               ):
    #set up some paths for the data
    dctMetaFiles['iTAKL'].index = ['_'.join(x.split('_')[1:]) for x in dctMetaFiles['iTAKL'].index.tolist()]

    #empty dataframe to add the meta for each subject to
    dfCombinedMeta = pd.DataFrame(columns = ['subject','scan','database','age','sex'])
    #fill out the meta dataframe with all subjects
    for strDatabase, strPath, strRating in zip(lDatabases, lPaths, lRatings):
        dfRating = pd.excel(strRating, index_col=0)
        for i, row in dfRating.iterrows():
            strSubject = row['strSubject']
            strSubjectPath = glob.glob(os.path.join(strPath, str(row['strSubject']),'*','ICA202DDisc'))[0]
            dctMeta = fGetMeta(strSubject, strDatabase)
            if strDatabase == 'HCP_MEG_20210212' and not pd.isnull(dctMeta['age']):
                dctMeta['age'] = np.mean([int(x) for x in dctMeta['age'].split('-')])
            dfCombinedMeta = dfCombinedMeta.append({'subject': strSubject,
                                                    'scan': row['strType'],
                                                    'database': strDatabase,
                                                    'age': dctMeta['age'],
                                                    'sex': dctMeta['sex']},
                                                    ignore_index=True)

    #assign a numeric subject ID, so a grouped split can be done
    intSubject = 0
    dfCombinedMeta['group'] = np.nan
    for i, row in dfCombinedMeta.iterrows():
        if pd.isnull(dfCombinedMeta.iloc[i]['group']):
            if row['database'] == 'iTAKL':
                strSubject = '_'.join(row['subject'].split('_')[-2:])
                lSubjectIndex = dfCombinedMeta[dfCombinedMeta['subject'].apply(lambda x: strSubject in str(x))].index
                if lSubjectIndex.__len__() == 1:
                    dfCombinedMeta.loc[i,'group'] = intSubject
                else:
                    dfCombinedMeta.loc[lSubjectIndex,'group'] = intSubject
            if row['database'] == 'HCP_MEG_20210212':
                lSubjectIndex = dfCombinedMeta[dfCombinedMeta['subject']==row['subject']].index
                if lSubjectIndex.__len__() == 1:
                    dfCombinedMeta.loc[i,'group'] = intSubject
                else:
                    dfCombinedMeta.loc[lSubjectIndex,'group'] = intSubject
            else:
                dfCombinedMeta.loc[i,'group'] = intSubject
            intSubject+=1

    #fill in the nan values of the meta so they can also be stratified
    dfCombinedMeta.fillna(-1, inplace=True)
    dfCombinedMeta['group'] = dfCombinedMeta['group'].astype(int)

    #drop the scans for replicated subjects, and split. 
    #Then use the lists of subjects to split the data (without grouping)
    #finally add back in the removed scans based on the split

    dfCombinedMetaSingleSubject = dfCombinedMeta.drop_duplicates('group').reset_index()

    lTrainDFs = []
    lValDFs = []
    lTestDFs = []

    #split the data
    #dfStratify = dfCombinedMeta[['scan','database','age','sex']]
    dfStratify = dfCombinedMetaSingleSubject[['age','sex', 'database']]

    splitTest = multi_stratified_k_fold(dfCombinedMetaSingleSubject,
                                       dfStratify,
                                       k=int(1/dTestFraction),
                                       lContinuousStratifiers = ['age'],
                                       seed=intRandomSeed,
                                       dRoundBase=1)

    for lTV, lTest in splitTest:
        #get the subjects for the test data
        dfTest = dfCombinedMetaSingleSubject.iloc[lTest]
        #get scans for all of the subjects
        dfTestScans = fBuildScanListsFromSubjectLists(dfTest, dfCombinedMeta)
        #add it to a list of dataframes
        lTestDFs.append(dfTestScans.reset_index(drop=True))
        #dataframe for the subjects in train and val data
        dfCombinedMetaSingleSubjectTV = dfCombinedMetaSingleSubject.iloc[lTV]
        #dataframe for the stratification of the subjects in the train and val data
        dfStratifyTV = dfStratify.iloc[lTV]

        #split the train and val data using k fold cross val
        splitVal = multi_stratified_k_fold(dfCombinedMetaSingleSubjectTV,
                                           dfStratifyTV,
                                           k=intK,
                                           lContinuousStratifiers = ['age'],
                                           seed=intRandomSeed,
                                           dRoundBase=1)
        intKCount = 0
        for lTrain, lVal in splitVal:
            print(lTrain.__len__(), lVal.__len__(), lTrain.__len__()+lVal.__len__())
            dfTrainFold = dfCombinedMetaSingleSubjectTV.iloc[lTrain]
            dfTrainFoldScans = fBuildScanListsFromSubjectLists(dfTrainFold, dfCombinedMeta)
            dfTrainFoldScans.columns = pd.MultiIndex.from_product([[intKCount],dfTrainFoldScans.columns])
            lTrainDFs.append(dfTrainFoldScans.reset_index(drop=True))
            dfValFold = dfCombinedMetaSingleSubjectTV.iloc[lVal]
            dfValFoldScan = fBuildScanListsFromSubjectLists(dfValFold, dfCombinedMeta)
            dfValFoldScan.columns = pd.MultiIndex.from_product([[intKCount],dfValFoldScan.columns])
            lValDFs.append(dfValFoldScan.reset_index(drop=True))
            intKCount += 1 
        break #not nestked k-fold, so stop on sinlge test split

    #these hold the subjects for each split
    dfTrain = pd.concat(lTrainDFs, axis=1)
    dfVal = pd.concat(lValDFs, axis=1)
    dfTest = pd.concat(lTestDFs, axis=1)

    dfTrain.to_csv('TrainSubjectSplit.csv')
    dfVal.to_csv('ValSubjectSplit.csv')
    dfTest.to_csv('TestSubjectSplit.csv')

    #check for leakage!
    lTVScans = dfTrain.loc[:,idx[:,'subject']].values.flatten().tolist()+dfVal.loc[:,idx[:,'subject']].values.flatten().tolist()
    lTVScans = list(set([x for x in lTVScans if not pd.isnull(x)]))
    lTestScans = dfTest.loc[:,idx['subject']]
    lTestScans = list(set([x for x in lTestScans if not pd.isnull(x)]))
    #ensure there's no test data in any train or val split
    assert([x for x in lTestScans if x in lTVScans].__len__()==0)
    #check lackage between each train and val split
    for i in range(intK):
        lTrainScans = dfTrain.loc[:,idx[i,'subject']].values.flatten().tolist()
        lTrainScans = list(set([x for x in lTrainScans if not pd.isnull(x)]))
        lValScans = dfVal.loc[:,idx[i,'subject']].values.flatten().tolist()
        lValScans = list(set([x for x in lValScans if not pd.isnull(x)]))
        assert([x for x in lValScans if x in lTrainScans].__len__()==0)

    #split stats
    lContinuousStratifiers = ['age']
    lCatagoricalStratifiers = ['sex','scan','database']
    dfSplitStats = pd.DataFrame(columns = lContinuousStratifiers+lCatagoricalStratifiers)
    dfCombinedMeta = dfCombinedMeta
    dfSplitMeta = dfTest
    dfSplitStats.loc['Test'] = fSplitStats(dfCombinedMeta, dfTest, lContinuousStratifiers, lCatagoricalStratifiers)
    for i in range(intK):
        dfSplitStats.loc[f'Train{i}'] = fSplitStats(dfCombinedMeta, dfTrain.loc[:,i], lContinuousStratifiers, lCatagoricalStratifiers)
        dfSplitStats.loc[f'Val{i}'] = fSplitStats(dfCombinedMeta, dfVal.loc[:,i], lContinuousStratifiers, lCatagoricalStratifiers)

    dfTrain.to_csv(os.path.join(strOutDir,'TrainScans.csv'))
    dfVal.to_csv(os.path.join(strOutDir,'ValidationScans.csv'))
    dfTest.to_csv(os.path.join(strOutDir,'TestScans.csv'))
    dfSplitStats.to_csv(os.path.join(strOutDir,'SplitStats.csv'))

    return dfTrain, dfVal, dfTest, dfSplitStats

def fGetStartTimesOverlap(intInputLen, intModelLen=15000, intOverlap=3750):
    """
    model len is 60 seconds at 250Hz = 15000
    overlap len is 15 seconds at 250Hz = 3750
    """
    lStartTimes = []
    intStartTime = 0
    while intStartTime+intModelLen<=intInputLen:
        lStartTimes.append(intStartTime)
        intStartTime = intStartTime+intModelLen-intOverlap
    return lStartTimes

def fChunkData(arrSpatialMap, arrTimeSeries, intLabel, intModelLen=15000, intOverlap=3750):
    intInputLen = arrTimeSeries.shape[0]
    lStartTimes = fGetStartTimesOverlap(intInputLen, intModelLen, intOverlap)

    lTemporalSubjectSlices = [arrTimeSeries[intStartTime:intStartTime+intModelLen] for intStartTime in lStartTimes]
    lSpatialSubjectSlices = [arrSpatialMap for intStartTime in lStartTimes]
    lLabel = [intLabel for intStartTime in lStartTimes]

    return lSpatialSubjectSlices, lTemporalSubjectSlices, lLabel


def fLoadAndPickleData(strTrainDFPath = os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit/TrainScans.csv'),
                       strValDFPath = os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit/ValidationScans.csv'),
                       strTestDFPath = os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit/TestScans.csv'),
                       dctDatabaseRatingPaths = {
                            'HCP_MEG_20210212':'/archive/bioinformatics/DLLab/AlexTreacher/data/MEGnet/ratings/RG_LB_AP_ED_HCP_MEG_20210212_final.xlsx',
                            'iTAKL':'/archive/bioinformatics/DLLab/AlexTreacher/data/MEGnet/ratings/RG_LB_AP_ED_iTAKL_final.xlsx',
                            'McGill':'/archive/bioinformatics/DLLab/AlexTreacher/data/MEGnet/ratings/RG_LB_AP_ED_McGill_final.xlsx',
                        },
                        intNComponents = 20,
                        strDataDir = os.path.join(paths.strDataDir, 'MEGnet/shared_ICA_raters'),
                        strOutDir = os.path.join(paths.strDataDir, 'MEGnet/FoldData'),
                        intModelLen = 60*250,
                        intOverlap=0,#for the training/validation there is no overlap set, for testing the complete lenght model (with voting) we will have a 15 second overlap
                        ):
    """

    Test data will not be chunked

    :param fortestingthecompletelenghtmodel: [description]
    :type fortestingthecompletelenghtmodel: [type]
    :param strTrainDFPath: [description], defaults to os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit/TrainScans.csv')
    :type strTrainDFPath: [type], optional
    :param strValDFPath: [description], defaults to os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit/ValidationScans.csv')
    :type strValDFPath: [type], optional
    :param strTestDFPath: [description], defaults to os.path.join(paths.strSourceDir, 'MEG_artifact/ExperimentOutputs/resubmission/DataSplit/TestScans.csv')
    :type strTestDFPath: [type], optional
    :param dctDatabaseRatingPaths: [description], defaults to { 'HCP_MEG_20210212':os.path.join(paths.strDataDir, 'MEGnet/shared_ICA_raters/RaterTemplateHCP_MEG_20210212_AT.csv'), 'iTAKL':os.path.join(paths.strDataDir, 'MEGnet/shared_ICA_raters/RaterTemplateiTAKL.csv'), 'McGill':os.path.join(paths.strDataDir, 'MEGnet/AlexMcGillRatings.csv') }
    :type dctDatabaseRatingPaths: dict, optional
    :param intNComponents: [description], defaults to 20
    :type intNComponents: int, optional
    :param strDataDir: [description], defaults to os.path.join(paths.strDataDir, 'MEGnet/shared_ICA_raters')
    :type strDataDir: [type], optional
    :param strOutDir: [description], defaults to os.path.join(paths.strDataDir, 'MEGnet/FoldData')
    :type strOutDir: [type], optional
    :param intModelLen: [description], defaults to 60*250
    :type intModelLen: [type], optional
    :param intOverlap: [description], defaults to 0
    :type intOverlap: int, optional
    """
    dctLabelConv = {'N':0, 'n':0, 0:0,
                    'B':1, 'b':1, 1:1,
                    'C':2, 'c':2, 2:2,
                    'S':3, 's':3, 3:3,
                    }
    
    dfTrainScans = pd.read_csv(strTrainDFPath, index_col=0, header=[0,1])
    dfValScans = pd.read_csv(strValDFPath, index_col=0, header=[0,1])
    dfTestScans = pd.read_csv(strTestDFPath, index_col=0)

    dctDataBaseRatings = dict([(strDB, pd.read_excel(strPath, index_col=0)) for strDB, strPath in dctDatabaseRatingPaths.items()])
    
    lTestTimeSeries = []
    lTestSpatialMap = []
    lTestLabel = []

    #save the test data as is (different lenghts)    
    for i, row in dfTestScans.iterrows():
        dfRatings = dctDataBaseRatings[row['database']]
        dfRatingsScan = dfRatings[(dfRatings['strSubject'].astype(str) == row['subject']) & (dfRatings['strType'] == row['scan'])]
        assert dfRatingsScan.shape[0] == 1
        pdsRatingsScan = dfRatingsScan.iloc[0]
        arrTimeSeries = loadmat(os.path.join(strDataDir, pdsRatingsScan['strPath'],'ICATimeSeries.mat'))['arrICATimeSeries']
        for intComp in range(1,intNComponents+1):
            arrSptatialMap = loadmat(os.path.join(strDataDir, pdsRatingsScan['strPath'],f'component{intComp}.mat'))['array'][30:-30,15:-15,:]
            lTestTimeSeries.append(arrTimeSeries[:,intComp-1])#min one as zero indexed
            lTestSpatialMap.append(arrSptatialMap)
            lTestLabel.append(dctLabelConv[pdsRatingsScan.loc[intComp]])
    for strPath, arr in zip(['lTestTimeSeries.pkl','lTestSpatialMap.pkl','lTestY.pkl'],
                            [lTestTimeSeries,lTestSpatialMap,lTestLabel]):
        #np.save(os.path.join(strOutDir, strPath), np.stack(arr))
        with open(os.path.join(strOutDir, strPath), 'wb') as f:
            pickle.dump(arr, f)

    #save the val data in 60s chunks
    for intFold in range(np.max([int(x[0]) for x in dfValScans.columns])+1):
        lTimeSeries = []
        lSpatialMap = []
        lLabels = []
        dfVal = dfValScans[str(intFold)].dropna()
        for i, row in dfVal.iterrows():
            dfRatings = dctDataBaseRatings[row['database']]
            dfRatingsScan = dfRatings[(dfRatings['strSubject'].astype(str) == row['subject']) & (dfRatings['strType'] == row['scan'])]
            assert dfRatingsScan.shape[0] == 1
            pdsRatingsScan = dfRatingsScan.iloc[0]
            arrTimeSeries = loadmat(os.path.join(strDataDir, pdsRatingsScan['strPath'],'ICATimeSeries.mat'))['arrICATimeSeries']
            for intComp in range(1,intNComponents+1):
                arrSpatialMap = loadmat(os.path.join(strDataDir, pdsRatingsScan['strPath'],f'component{intComp}.mat'))['array'][30:-30,15:-15,:]
                #Split the temporal signal into 60 second chunks with 0 seconds overlap
                lSpatialSubjectSlices, lTemporalSubjectSlices, lLabelSubject = fChunkData(arrSpatialMap, 
                                                                                          arrTimeSeries[:,intComp-1], 
                                                                                          dctLabelConv[pdsRatingsScan.loc[intComp]],
                                                                                          intModelLen=intModelLen,
                                                                                          intOverlap=intOverlap)
                lTimeSeries.extend(lTemporalSubjectSlices)
                lSpatialMap.extend(lSpatialSubjectSlices)
                lLabels.extend(lLabelSubject)
        for strPath, arr in zip([f'arrValidation{intFold}TimeSeries.npy',f'arrValidation{intFold}SpatialMap.npy',f'arrValidation{intFold}Y.npy'],
                                [lTimeSeries,lSpatialMap,lLabels]):
            np.save(os.path.join(strOutDir, strPath), np.stack(arr))
            #with open(os.path.join(strOutDir, strPath), 'wb') as f:
            #    pickle.dump(arr, f)
        
    #save the train data in 60s chunks
    for intFold in range(np.max([int(x[0]) for x in dfTrainScans.columns])+1):
        lTimeSeries = []
        lSpatialMap = []
        lLabels = []
        dfTrain = dfTrainScans[str(intFold)].dropna()
        for i, row in dfTrain.iterrows():
            dfRatings = dctDataBaseRatings[row['database']]
            dfRatingsScan = dfRatings[(dfRatings['strSubject'].astype(str) == row['subject']) & (dfRatings['strType'] == row['scan'])]
            assert dfRatingsScan.shape[0] == 1
            pdsRatingsScan = dfRatingsScan.iloc[0]
            arrTimeSeries = loadmat(os.path.join(strDataDir, pdsRatingsScan['strPath'],'ICATimeSeries.mat'))['arrICATimeSeries']
            for intComp in range(1,intNComponents+1):
                arrSpatialMap = loadmat(os.path.join(strDataDir, pdsRatingsScan['strPath'],f'component{intComp}.mat'))['array'][30:-30,15:-15,:]
                #Split the temporal signal into 60 second chunks with 0 seconds overlap
                lSpatialSubjectSlices, lTemporalSubjectSlices, lLabelSubject = fChunkData(arrSpatialMap, 
                                                                                          arrTimeSeries[:,intComp-1], 
                                                                                          dctLabelConv[pdsRatingsScan.loc[intComp]],
                                                                                          intModelLen=intModelLen,
                                                                                          intOverlap=intOverlap)
                lTimeSeries.extend(lTemporalSubjectSlices)
                lSpatialMap.extend(lSpatialSubjectSlices)
                lLabels.extend(lLabelSubject)
        for strPath, arr in zip([f'arrTrain{intFold}TimeSeries.npy',f'arrTrain{intFold}SpatialMap.npy',f'arrTrain{intFold}Y.npy'],
                                [lTimeSeries,lSpatialMap,lLabels]):
            np.save(os.path.join(strOutDir, strPath), np.stack(arr))
            #with open(os.path.join(strOutDir, strPath), 'wb') as f:
            #    pickle.dump(arr, f)

def fPredictChunkAndVoting(kModel, lTimeSeries, arrSpatialMap, arrY, intModelLen=15000, intOverlap=3750):
    """
    This function is designed to take in ICA time series and a spatial map pair and produce a prediction useing a trained model.
    The time series will be split into multiple chunks and the final prediction will be a weighted vote of each time chunk.
    The weight for the voting will be determined by the manout of time and overlap each chunk has with one another.
    For example if the total lenght of the scan is 50 seconds, and the chunks are 15 seconds long with a 5 second overlap:
        The first chunk will be the only chunk to use the first 10 seconds, and one of two chunks to use the next 5 seconds.
            Thus   

    :param kModel: The model that will be used for the predictions on each chunk. It should have two inputs the spatial map and time series respectivley
    :type kModel: a keras model
    :param lTimeSeries: The time series for each scan (can also be an array if all scans are the same lenght)
    :type lTimeSeries: list or array (if each scan is a different length, then it needs to be a list)
    :param arrSpatialMap: The spatial maps (one per scan)
    :type arrSpatialMap: numpy array
    :param intModelLen: The lenght of the time series in the model, defaults to 15000
    :type intModelLen: int, optional
    :param intOverlap: The lenght of the overlap between scans, defaults to 3750
    :type intOverlap: int, optional
    """
    #empty list to hold the prediction for each component pair
    lPredictionsVote = []
    lGTVote = []

    lPredictionsChunk = []
    lGTChunk = []

    i = 0
    for arrScanTimeSeries, arrScanSpatialMap, arrScanY in zip(lTimeSeries, arrSpatialMap, arrY):
        intTimeSeriesLen = arrScanTimeSeries.shape[0]
        lStartTimes = fGetStartTimesOverlap(intTimeSeriesLen, intModelLen=intModelLen, intOverlap=intOverlap)

        if lStartTimes[-1]+intModelLen <= intTimeSeriesLen:
            lStartTimes.append(arrScanTimeSeries.shape[0]-intModelLen)


        lTimeChunks = [[x,x+intModelLen] for x in lStartTimes]
        dctTimeChunkVotes = dict([[x,0] for x in lStartTimes])
        for intT in range(intTimeSeriesLen):
            lChunkMatches = [x <= intT < x+intModelLen for x in dctTimeChunkVotes.keys()]
            intInChunks = np.sum(lChunkMatches)
            for intStartTime, bTruth in zip(dctTimeChunkVotes.keys(), lChunkMatches):
                if bTruth:
                    dctTimeChunkVotes[intStartTime]+=1.0/intInChunks

        #predict
        dctWeightedPredictions = {}
        for intStartTime in dctTimeChunkVotes.keys():
            lPrediction = kModel.predict([np.expand_dims(arrScanSpatialMap,0),
                                        np.expand_dims(np.expand_dims(arrScanTimeSeries[intStartTime:intStartTime+intModelLen],0),-1)])
            lPredictionsChunk.append(lPrediction)
            lGTChunk.append(arrScanY)
            
            dctWeightedPredictions[intStartTime] = lPrediction*dctTimeChunkVotes[intStartTime]

        arrScanPrediction = np.stack(dctWeightedPredictions.values())
        arrScanPrediction = arrScanPrediction.mean(axis=0)
        arrScanPrediction = arrScanPrediction/arrScanPrediction.sum()
        lPredictionsVote.append(arrScanPrediction)
        lGTVote.append(arrScanY)
        
        #print(f"{i}/{arrY.shape[0]}")
        i+=1
    return np.stack(lPredictionsVote), np.stack(lGTVote), np.stack(lPredictionsChunk), np.stack(lGTChunk)

def fReadConfig(strPath):
    """Reads a config from a text file and returns the config dict

    :param strPath: [description]
    :type strPath: [type]
    :return: [description]
    :rtype: [type]
    """
    if os.path.exists(strPath):
        with open(strPath, 'r') as f:
            dctConfig = json.loads(f.readlines()[0])
    else:
        dctConfig = None
    return dctConfig

def fGetModelStatsFromTrainingHistory(strPath):
    """Get the summary stats from the training history across folds
    Input dataframe should have a multiindex for columns. 
        0. The fold
        1. The performance metric

    :param strPath: path to history dataframe
    :type strPath: str
    :return: summary values across folds
    :rtype: dct
    """
    dfHistory = pd.read_csv(strPath, index_col=0, header=[0,1])
    dfValF1 = dfHistory.loc[:, idx[:,'val_f1_score']]
    dctReturn = {'mean_val_f1': dfValF1.max().mean(),
                 'std_val_f1': dfValF1.max().std(),
                 'mean_epochs': dfValF1.idxmax().mean()}
    return pd.Series(dctReturn)

def fSummaryDFFromRS(strRSRoot):
    """Creates a summary dataframe from the RS that includes configs and performance.
    Useful to determine best model, and for HPO analysis

    :param strRSRoot: path to the root folder that contains the RS models
    :type strRSRoot: str
    :return: pandas dataframe of the RS summary
    :rtype: pd.DataFrame
    """
    lTrainingHistoryPaths = glob.glob(os.path.join(strRSRoot,'*','training_history.csv'))
    lTrainingHistoryPaths.sort(key = lambda x: int(x.split(os.sep)[-2][5:]))
    lDFTrainingHistory = [pd.read_csv(x, index_col=0, header=[0,1]) for x in lTrainingHistoryPaths]
    lF1Val = [x.loc[:, idx[:,'val_f1_score']] for x in lDFTrainingHistory]
    dfSummary = pd.DataFrame(columns = ['model_path','model_num','mean_val_f1','std_val_f1','mean_epochs'])
    dfSummary['model_path'] = [os.path.dirname(x) for x in lTrainingHistoryPaths]
    dfSummary['model_num'] = dfSummary['model_path'].apply(lambda x: x.split(os.sep)[-1])
    dfSummary['config'] = [fReadConfig(os.path.join(x, 'config.txt')) for x in dfSummary['model_path']]

    dfSummary[['mean_val_f1','std_val_f1','mean_epochs']] = (dfSummary['model_path']+'/training_history.csv').apply(fGetModelStatsFromTrainingHistory)
    dfSummary['95CI_val_f1'] = dfSummary['mean_val_f1']-1.96*dfSummary['std_val_f1']
    return dfSummary
