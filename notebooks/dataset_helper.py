import os
from pathlib import Path
import types

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from math import ceil
from sklearn.metrics import roc_curve, auc

import sys

from codetector.src.features.shared.domain.entities.samples.detection_sample import DetectionSample
#To import modules
sys.path.append('../')
from codetector.src.features.shared.data.models.code_detection_sample_model import CodeDetectionSampleModel
from codetector.src.features.shared.domain.entities.dataset.dataset import Dataset
from codetector.src.features.shared.domain.entities.tokenizer import Tokenizer


class DatasetHelper(object):
    """
    Utility class that works with Dataset objects.
    """


    def calculateAUROCScores(self, dataset: Dataset, df:DataFrame=None, sameGeneratorOnly:bool=True, baseModelOverride:str=None, generatorOverride:str=None, secondaryModelOverride:str=None, flipList:list[str]=[], returnFprTpr:bool=False, loadingBar:bool=False) -> dict:
        """
        Calculate the AUROC Scores of this dataset. Returns a dictionary with `{"detector": {"basemodel": auroc, ...}, ...}`.
        dataset: The dataset to be analyzed.
        df: If Pandas DataFrame variant already exists, then pass that over as well to speed up computation.
        sameGeneratorOnly: Only compare detection values of samples with the same generator as the base model in the detector.
        baseModelOverride: If not None, only that base model will be considered in the AUROC.
        generatorOverride: If not None, only that generator will be considered in the AUROC.
        flipList: List of detectors that should be inverted.
        returnFprTpr: When set to true, return 3-tuple of auroc,fpr,tpr.
        loadingBar: Show a loading bar.
        """

        assert issubclass(dataset.getContentType(), DetectionSample)

        #https://thomas-cokelaer.info/blog/2014/06/pandas-how-to-compare-dataframe-with-none/
        df = dataset.toDataframe() if isinstance(df, types.NoneType) else df


        bar = tqdm(desc='Calculating AUROC',total=df.count()['Value']) if loadingBar else None
        total = 0

        detectors : list[str] = list(df['Detector'].value_counts().to_dict().keys())
        # if loadingBar:
        #     detectors = tqdm(detectors, desc='Calculating AUROC for detectors', position=1)

        baseModels : list[str] = list(df['BaseModel'].value_counts().to_dict().keys()) if baseModelOverride == None else [baseModelOverride]

        # if loadingBar:
        #     baseModels = tqdm(baseModels, desc='Calculating AUROC for base models', position=2, leave=False)

        real : dict[str,dict[str,list]] = {}
        fake : dict[str,dict[str,list]] = {}

        for detector in detectors:
            #Initialize lists for ROC
            if not (detector in real):
                real[detector] = {}
                fake[detector] = {}

            #https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
            detectorFilteredDF = df.loc[df['Detector'] == detector]

            for baseModel in baseModels:


                if not(baseModel) in real[detector]:                      
                    real[detector][baseModel] = []

                if not(baseModel) in fake[detector]:                      
                    fake[detector][baseModel] = []


                baseModelFilteredDF = detectorFilteredDF.loc[detectorFilteredDF['BaseModel'] == baseModel]

                if generatorOverride:
                    #https://www.statology.org/or-operator-in-pandas/
                    baseModelFilteredDF = baseModelFilteredDF.loc[(baseModelFilteredDF['Generator'] == generatorOverride) | (baseModelFilteredDF['Generator'] == 'human')]

                if sameGeneratorOnly:
                    baseModelFilteredDF = baseModelFilteredDF.loc[(baseModelFilteredDF['Generator'] == baseModel) | (baseModelFilteredDF['Generator'] == 'human')]

            
                if secondaryModelOverride == 'None':
                    #https://saturncloud.io/blog/python-pandas-selecting-rows-whose-column-value-is-null-none-nan/
                    baseModelFilteredDF = baseModelFilteredDF.loc[baseModelFilteredDF['SecondaryModel'].isnull()]
                elif secondaryModelOverride:
                    baseModelFilteredDF = baseModelFilteredDF.loc[(baseModelFilteredDF['SecondaryModel'] == secondaryModelOverride)]

                rawSamples = baseModelFilteredDF.to_dict('records')

                # if loadingBar:
                #     rawSamples = tqdm(rawSamples,desc='Calculating AUROC for filtered samples',position=3,leave=False)

                if loadingBar:
                    total += len(rawSamples)
                    bar.total = total

                for detectionSample in rawSamples:

                    if loadingBar:
                        bar.update()

                    #Sample type must implement MappableMixin !!!
                    sample : DetectionSample = dataset.getContentType().fromDict(detectionSample)

                    if dataset.passesFilters(sample):
                        if sample.generatorTag != 'human':
                            val = sample.value
                            #Always pessimistic
                            if np.isnan(val):
                                val = -100000
                            elif np.isinf(val):
                                tempval = 100000
                                val = -tempval if np.isneginf(val) else tempval
                            
                            val = -val if detector in flipList else val
                            fake[detector][baseModel].append(val)
                        else:
                            val = sample.value
                            #Always pessimistic
                            if np.isnan(val):
                                val = 100000
                            elif np.isinf(val):
                                tempval = 100000
                                val = -tempval if np.isneginf(val) else tempval


                            val = -val if detector in flipList else val

                            real[detector][baseModel].append(val)

        output : dict[str,dict[str,float]] = {}

        for detector in real.keys():
            output[detector] = {}
            for baseModel in real[detector].keys():

                #Skip ignored values
                if len(real[detector][baseModel]) == 0 or len(fake[detector][baseModel]) == 0:
                    continue

                fpr, tpr, thresh = roc_curve([0] * len(real[detector][baseModel]) + [1] * len(fake[detector][baseModel]), real[detector][baseModel] + fake[detector][baseModel])
                roc_auc = float(auc(fpr, tpr))
                output[detector][baseModel] = roc_auc if not returnFprTpr else (roc_auc,fpr.tolist(),tpr.tolist(), thresh.tolist())
        
        return output


    def generateTokenFrequencies(self, dataset:Dataset|DataFrame, tokenizer:Tokenizer, columnName:str='Code') -> dict[str,int]:
        """
        Given a dataset return a dictionary containing all token frequencies.
        """

        #https://stackoverflow.com/questions/722697/best-way-to-turn-word-list-into-frequency-dict

        df = dataset.toDataframe() if not isinstance(dataset, DataFrame) else dataset

        codeList = df[columnName].values

        batchSize = 120

        tokenFreq : dict[int,int] = {}

        with tqdm(total=ceil(len(codeList)/batchSize),desc='Counting tokens') as bar:
            index = 0
            batch = list(map(lambda x : str(x), codeList[index*batchSize:(index+1)*batchSize]))
            while len(batch) != 0:
                tokens = tokenizer.encodeBatch(batch)

                for codeTokens in tokens:
                    freq = {i:codeTokens.count(i) for i in set(codeTokens)}
                    for key,val in freq.items():
                        if not(key in tokenFreq):
                            tokenFreq[key] = 0
                        tokenFreq[key]+=val

                index+= 1
                batch = list(map(lambda x : str(x), codeList[index*batchSize:(index+1)*batchSize]))
                bar.update()

        finalFreq = {}
        for key,val in tokenFreq.items():
            finalFreq[tokenizer.decode([key])] = val       


        return finalFreq
    


    def generateHeatmap(self,
                        dataset:Dataset,
                        df:DataFrame=None,
                        generators:list[str]=None,
                        baseModels:list[str]=None,
                        detectors:list[str]=None,
                        folderPath:str=None,
                        suffix:str='',
                        splitSecondary:bool=False,
                        flipList:list[str]=[]) -> None:
        """
        Generate figures of detectors performance with cross model/generator.
        dataset: The dataset to generate heatmaps from.
        generators: List of generator names to compare.
        baseModels: List of base models to compare.
        detectors: List of names of detection methods to compare.
        flipList: List of detectors to invert.
        """
        if generators == None or baseModels == None or detectors == None:
            return
        
        assert issubclass(dataset.getContentType(), DetectionSample)

        df = dataset.toDataframe() if isinstance(df, types.NoneType) else df

        secondaryModels = ['None'] + list(baseModels) if splitSecondary else [None]

        for secondaryModel in secondaryModels:

            data :dict[str,np.ndarray] = {}

            for detector in detectors:
                data[detector] = np.zeros([len(generators),len(baseModels)],dtype=np.float32)
            
            data['max'] = np.zeros([len(generators),len(baseModels)],dtype=np.float32)


            

            with tqdm(total=len(generators)*len(baseModels), desc='Calculating AUROC for heatmaps', position=0) as bar:
                for generatorIndex, generator in enumerate(generators):

                    for baseModelIndex, baseModel in enumerate(baseModels):
                        auroc = self.calculateAUROCScores(dataset, df=df, sameGeneratorOnly=False,baseModelOverride=baseModel,generatorOverride=generator,secondaryModelOverride=secondaryModel)             
                        bar.update()

                        for detector in auroc.keys():
                            if not detector in data:
                                continue
                            for baseModel in auroc[detector]:
                                data[detector][generatorIndex,baseModelIndex] = auroc[detector][baseModel]
                                data['max'][generatorIndex,baseModelIndex] = max(auroc[detector][baseModel], data['max'][generatorIndex,baseModelIndex])



            if folderPath:
                #https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

                for detector in data.keys():

                    fig, ax = plt.subplots(figsize=(18,15))

                    vMax = data[detector].max() if not detector in flipList else 1 - data[detector].min()
                    values = data[detector] if not detector in flipList else 1 - data[detector]

                    #https://stackoverflow.com/questions/52758070/color-map-to-shades-of-blue-python
                    #https://matplotlib.org/stable/users/explain/colors/colormapnorms.html
                    norm = mpl.colors.Normalize(vmin=0, vmax=vMax)


                    heatmap = ax.imshow(values,cmap=mpl.cm.Blues, norm=norm)
                    # fig.colorbar(heatmap)

                    #https://www.geeksforgeeks.org/how-to-set-tick-labels-font-size-in-matplotlib/
                    ax.set_xticks(range(len(baseModels)), labels=baseModels,fontsize=50/np.sqrt(len(values)))
                    ax.set_yticks(range(len(generators)),labels=generators,fontsize=50/np.sqrt(len(values)))


                    ax.tick_params(top=True, bottom=False,
                            labeltop=True, labelbottom=False)

                    # ax.set_title(detector, fontsize=90/np.sqrt(len(values)), y=-0.11) #, labelpad=7

                    #https://stackoverflow.com/questions/6406368/move-x-axis-label-downwards-but-not-x-axis-ticks-in-matplotlib
                    ax.set_xlabel('Detection Model', fontsize=48, labelpad=35) #80/np.sqrt(len(values))
                    ax.set_ylabel('Generator Model', fontsize=48, labelpad=35) #80/np.sqrt(len(values))

                    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                            rotation_mode="anchor")
                    
                    for i in range(len(generators)):
                        for j in range(len(baseModels)):
                            #https://stackoverflow.com/questions/33104322/auto-adjust-font-size-in-seaborn-heatmap
                            text = ax.text(j, i, round(values[i, j],2),
                                        ha="center", va="center", color="w" if abs(round(values[i, j],2)) > 0.05 else 'black',size=60/np.sqrt(len(values)))


                    if secondaryModel != None:
                        folderPath += f'/bySecondary/{secondaryModel}'
                        if not Path(folderPath).exists():
                            os.mkdir(folderPath)

                    plt.margins(x=0, y=0)
                    # plt.rcParams['svg.fonttype'] = 'none'
                    plt.rcParams.update({
                        'svg.fonttype': 'none',
                        'font.family': 'serif',
                        'font.serif': ['CMU Serif Roman', 'CMU Serif'],
                        'mathtext.fontset': 'cm',
                        'xtick.labelsize': 22,
                        'ytick.labelsize': 22,
                        'axes.labelsize': 48,
                    })
                    plt.tick_params(axis='both', which='major', labelsize=22)

                    # plt.rcParams.update({
                    #     'svg.fonttype': 'none',
                    #     'font.family': 'serif',
                    #     'font.serif': ['Computer Modern Roman'],
                    #     'mathtext.fontset': 'cm'
                    # })
                    plt.savefig(f'{folderPath}/{detector}{suffix}.png',dpi=100)#/animation
                    plt.savefig(f'{folderPath}/{detector}{suffix}.svg')
                    del ax,fig
                    plt.close()



    def generateDifferenceHeatmap(self, datasets:tuple[Dataset,Dataset],
                                  df:tuple[DataFrame,DataFrame]=None,
                                  generators:list[str]=None,
                                  baseModels:list[str]=None,
                                  detectors:list[str]=None,
                                  folderPath:str=None,
                                  suffix:str='',
                                  flipList:list[str]=[],
                                  diagonalDiff:bool=False) -> dict[str,dict[str,list[float]]]:
        """
        Generate difference heatmap using two datasets with cross model/generator. Calculates datasets[0]-datasets[1]
        dataset: The dataset pair to generate heatmaps from.
        df: The dataframe pair if they exist.
        generators: List of generator names to compare.
        baseModels: List of base models to compare.
        detectors: List of names of detection methods to compare.
        flipList: List of detectors to invert.
        diagonalDiff: Highlight the difference to the diagonal (white box).

        Returns a dictionary with the auroc scores of the detector across both datasets `{'<detector>': {'comparison': [0.6,0.7,...], 'auroc': [0.5,0.6,...]}, ...}`.
        """
        if generators == None or baseModels == None or detectors == None:
            return
        
        assert len(datasets) == 2 and (df == None or len(df) == 2), 'Invalid dataset pair'

        assert issubclass(datasets[0].getContentType(), DetectionSample)
        assert issubclass(datasets[1].getContentType(), DetectionSample)

        


        comparisonDf = datasets[0].toDataframe() if isinstance(df[0], types.NoneType) else df[0]
        df = datasets[1].toDataframe() if isinstance(df[1], types.NoneType) else df[1]


        data :dict[str,np.ndarray] = {}

        for detector in detectors:
            data[detector] = np.zeros([len(generators),len(baseModels)],dtype=np.float32)
        
        data['max'] = np.zeros([len(generators),len(baseModels)],dtype=np.float32)


        aurocToReturn : dict[str,dict[str,list[float]]] = {}
        

        with tqdm(total=len(generators)*len(baseModels), desc='Calculating AUROC difference heat maps', position=0) as bar:
            for generatorIndex, generator in enumerate(generators):
                for baseModelIndex, baseModel in enumerate(baseModels):
                    auroc = self.calculateAUROCScores(datasets[1], df=df, sameGeneratorOnly=False,baseModelOverride=baseModel,generatorOverride=generator)
                    aurocComparison = self.calculateAUROCScores(datasets[0], df=comparisonDf, sameGeneratorOnly=False,baseModelOverride=baseModel,generatorOverride=generator)

                    bar.update()
                    for detector in auroc.keys():
                        for baseModel in auroc[detector]:
                            data[detector][generatorIndex,baseModelIndex] = aurocComparison[detector][baseModel] - auroc[detector][baseModel]

                            if not(detector in aurocToReturn):
                                aurocToReturn[detector] = {'comparison':[], 'auroc':[]}

                            aurocToReturn[detector]['comparison'].append(aurocComparison[detector][baseModel])
                            aurocToReturn[detector]['auroc'].append(auroc[detector][baseModel])

                            temp = aurocComparison[detector][baseModel] - auroc[detector][baseModel]
                            data['max'][generatorIndex,baseModelIndex] = temp if abs(temp) > abs(data['max'][generatorIndex,baseModelIndex]) else data['max'][generatorIndex,baseModelIndex]
      



        if folderPath:
            #https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

            for detector in data.keys():

                #https://stackoverflow.com/questions/14770735/how-do-i-change-the-figure-size-with-subplots
                fig, ax = plt.subplots(figsize=(18,15))


                vMax = data[detector].max() if not detector in flipList else - data[detector].min()
                vMin = data[detector].min() if not detector in flipList else - data[detector].max()
                values = data[detector] if not detector in flipList else - data[detector]

                if diagonalDiff:
                    with tqdm(total=len(generators)*len(baseModels), desc='Subtracting diagonal values', position=0) as bar:
                        for generatorIndex, generator in enumerate(generators):
                            for baseModelIndex, baseModel in enumerate(baseModels):
                                #Skip diagonal for now, otherwise values will be incorrect
                                if baseModelIndex == generatorIndex:
                                    continue
                                values[generatorIndex,baseModelIndex] -= (values[generatorIndex,generatorIndex] + values[baseModelIndex,baseModelIndex])/2.0
                                bar.update()
                        #Set diagonal to zero
                        for generatorIndex, generator in enumerate(generators):
                            for baseModelIndex, baseModel in enumerate(baseModels):
                                if baseModelIndex == generatorIndex:
                                    values[generatorIndex,baseModelIndex] = 0

                #https://stackoverflow.com/questions/52758070/color-map-to-shades-of-blue-python
                #https://matplotlib.org/stable/users/explain/colors/colormapnorms.html
                #https://stackoverflow.com/questions/38246559/how-to-create-a-heat-map-in-python-that-ranges-from-green-to-red
                norm = mpl.colors.Normalize(vmin=-1, vmax=1)

                c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
                v = [0,.15,.4,.5,0.6,.9,1.]
                l = list(zip(v,c))
                cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

                heatmap = ax.imshow(values,cmap=cmap, norm=norm)
                # fig.colorbar(heatmap)

                #https://www.geeksforgeeks.org/how-to-set-tick-labels-font-size-in-matplotlib/
                ax.set_xticks(range(len(baseModels)), labels=baseModels,fontsize=50/np.sqrt(len(values)))
                ax.set_yticks(range(len(generators)),labels=generators,fontsize=50/np.sqrt(len(values)))


                ax.tick_params(top=True, bottom=False,
                        labeltop=True, labelbottom=False)

                # ax.set_title(detector, fontsize=90/np.sqrt(len(values)), y=-0.11) #, labelpad=7

                #https://stackoverflow.com/questions/6406368/move-x-axis-label-downwards-but-not-x-axis-ticks-in-matplotlib
                ax.set_xlabel('Detection Model', fontsize=48, labelpad=35) #80/np.sqrt(len(values))
                ax.set_ylabel('Generator Model', fontsize=48, labelpad=35) #80/np.sqrt(len(values))

                plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                        rotation_mode="anchor")
                
                for i in range(len(generators)):
                    for j in range(len(baseModels)):
                        #https://stackoverflow.com/questions/33104322/auto-adjust-font-size-in-seaborn-heatmap
                        text = ax.text(j, i, round(values[i, j],2),
                                    ha="center", va="center", color="w" if abs(round(values[i, j],2)) > 0.05 else 'black',size=60/np.sqrt(len(values)))

                plt.margins(x=0, y=0)

                newSuffix = suffix
                if diagonalDiff:
                    newSuffix = f'{suffix}_diag'


                plt.rcParams.update({
                        'svg.fonttype': 'none',
                        'font.family': 'serif',
                        'font.serif': ['CMU Serif Roman', 'CMU Serif'],
                        'mathtext.fontset': 'cm',
                        'xtick.labelsize': 22,
                        'ytick.labelsize': 22,
                        'axes.labelsize': 48,
                    })
                plt.tick_params(axis='both', which='major', labelsize=22)

                plt.savefig(f'{folderPath}/{detector}_difference{newSuffix}.png',dpi=100)#/animation
                plt.savefig(f'{folderPath}/{detector}_difference{newSuffix}.svg')
                del ax,fig
                plt.close()


        return aurocToReturn