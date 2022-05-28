import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import kmeans
import spectralClustering
import meanShift
import dbscan
import optics
import affinityPropagation
import fuzzyKMeans
import pandas as pd
import elbowKmeans
import elbowKmeansPl
from algorithms.patternmining.createDB import createDB
from algorithms.patternmining.euclidDistance import EuclidDistance
from dataProcessing.VerticalExpansion import verticalExpansion
from dataProcessing.HorizontalExpansion import HorizontalExpansion
import periodicFrequentPattern
# from Imputation_methods import *
import os
import math
from histif.src.main_psf import main_psf
import uuid
import numpy as np
from osgeo import gdal
from oneclassClassifiers import *
# from histif.src.


class GUImain:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RasterMiner: Discovering Knowledge Hidden in Raster Images")
        self.root.minsize(600, 400)

    def uploadInputDir(self, event=None):
        # filename = filedialog.askopenfilename()
        inputDir = filedialog.askdirectory()
        # print('Selected:', inputDir)
        inputRasterFolder = inputDir
        return inputRasterFolder

    def uploadOutputDir(self, event=None):
        # filename = filedialog.askopenfilename()
        outDir = filedialog.askdirectory()  
        # print('Selected:', outDir)
        outputFolder = outDir
        return outputFolder

    def uploadInputFile(self):
        inputFile = filedialog.askopenfilename()
        # print('selected:', inputFile)
        return inputFile

    def uploadOutputFile(self):
        outputFile = filedialog.askopenfilename()
        # print('selected:', outputFile)
        return outputFile

    def rasterToHorizontal(self, inputRasterFolderName, fileExtension, outputFolderName, startBandVar, endBandVar):
        print("Calling HorizontalExpansion.py")
        a = HorizontalExpansion(inputRasterFolderName, fileExtension, outputFolderName,
                                startBandVar, endBandVar)
        a.convert()
        statusVar.set(str("Status : Preprocessing Completed !"))
        iFileNameHandlingNan.set(str(outputFolderName) + '/rawData.tsv')

        print("completed")

        #     print(f'-bounds{i}',end=' ')

    def rasterToVertical(self, inputRasterFolderName, fileExtension, outputFolderName):
        print("Calling VerticalExpansion.py")
        a = verticalExpansion(inputRasterFolderName, fileExtension, outputFolderName)
        a.convert()
        statusVar.set(str("Status : Preprocessing Completed !"))
        iFileNameHandlingNan.set(str(outputFolderName) + '/rawData.tsv')

    def patternMiningTempFile(self, inputFileName, outputFileName, condition, threshold):
        createDB(inputFileName, outputFileName
                 , condition, int(threshold)).run()
        statusVar.set(str("Status : Pattern Mining Completed !"))

    def patternMiningNeighbourFile(self, inputFileName, outputFileName, threshold):
        EuclidDistance(inputFileName, outputFileName, threshold).run()

        statusVar.set(str("Status : Pattern Mining Completed !"))

    def HISTIF(self, fineImaget0InputFile, coarseImaget0InputFile, coarseImaget1InputFile, imageFusionOutputFolderName, neighbors, params, iterations, variantType):

        print("Calling HISTIF.py")
        
        old_version = False
        print(variantType)
        if(variantType == 2):
            old_version = True

        img_data_new, final_img = main_psf(coarse_img_t0 = coarseImaget0InputFile, fine_img_t0 = fineImaget0InputFile, coarse_img_t1 = coarseImaget1InputFile, params = params, iter = iterations, neighbors = neighbors, old_version = old_version)

        dst_filename = str(uuid.uuid4())
        x_pixels = (final_img[0]).shape[0]  # number of pixels in x
        y_pixels = (final_img[0]).shape[1]  # number of pixels in y
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(f'{imageFusionOutputFolderName}/{dst_filename}.tif',x_pixels, y_pixels, len(final_img),gdal.GDT_Float32)
        
        for band_range in range(len(final_img)):
            dataset.GetRasterBand(band_range+1).WriteArray(final_img[band_range])
        
        data0 = gdal.Open(fineImaget0InputFile)
        geotrans=data0.GetGeoTransform()
        proj=data0.GetProjection()
        dataset.SetGeoTransform(geotrans)
        dataset.SetProjection(proj)
        dataset.FlushCache()
        dataset=None
        statusVar.set(str("Status : Image Fusion Completed !"))

    def recommender(self, iTrainFile, iTestFile, outputFolder, topKValue):
        if oneNNEDFlag.get():
            train_df = pd.read_csv(iTrainFile, header=None, sep='\t')
            test_df = pd.read_csv(iTestFile, header=None, sep='\t')
            FinalSamples, TopKSamples = rasterOneNNED(train_df, test_df, int(topKValue))
            saveFinalTestSamples(FinalSamples, outputFolder + str("/oneNNEDFinalSamples.csv"))
            saveFinalTestSamples(TopKSamples, outputFolder+str("/oneNNEDTopKSamples.csv"))

        if oneNNDTWFlag.get():
            train_df = pd.read_csv(iTrainFile, header=None, sep='\t')
            test_df = pd.read_csv(iTestFile, header=None, sep='\t')
            FinalSamples, TopKSamples = rasterOneNNDTW(train_df, test_df, int(topKValue))
            saveFinalTestSamples(FinalSamples, outputFolder + str("/oneNNDTWFinalSamples.csv"))
            saveFinalTestSamples(TopKSamples, outputFolder+str("/oneNNDTWTopKSamples.csv"))

        if oneNNHausdorffFlag.get():
            train_df = pd.read_csv(iTrainFile, header=None, sep='\t')
            test_df = pd.read_csv(iTestFile, header=None, sep='\t')
            FinalSamples, TopKSamples = rasterOneNNHausdorff(train_df, test_df, int(topKValue))
            saveFinalTestSamples(FinalSamples, outputFolder + str("/oneNNHausdorffFinalSamples.csv"))
            saveFinalTestSamples(TopKSamples, outputFolder+str("/oneNNHausdorffTopKSamples.csv"))

        if oneNNmaxinormFlag.get():
            train_df = pd.read_csv(iTrainFile, header=None, sep='\t')
            test_df = pd.read_csv(iTestFile, header=None, sep='\t')
            FinalSamples, TopKSamples = rasterOneNNMaxNorm(train_df, test_df, int(topKValue))
            saveFinalTestSamples(FinalSamples, outputFolder + str("/oneNNmaxinormFinalSamples.csv"))
            saveFinalTestSamples(TopKSamples, outputFolder+str("/oneNNmaxinormTopKSamples.csv"))


        statusVar.set(str("Status : Recommender System Completed !"))
        messagebox.showinfo('notification', 'Successfully completed')

            # pass

    def judgeClusteringAlg(self, target):
        self.root.destroy()
        if target == 'k-Means/k-Means++':
            kmeans.kmeansGUI().Main()
        elif target == 'fuzzy-kMeans':
           fuzzyKMeans.fuzzyKMeansGUI().Main()
        elif target == 'DBScan':
            dbscan.DBScanGUI().Main()
        elif target == 'MeanShift':
            meanShift.meanShiftGUI().Main()
        elif target == 'SpectralClustering':
            spectralClustering.spectralGUI().Main()
        elif target == 'OPTICS':
            optics.opticsGUI().Main()
        elif target == 'AffinityPropagation':
            affinityPropagation.affinityPropagationGUI().Main()
        elif target == 'Elbow-kmeans':
            elbowKmeans.elbowKmeansGUI().Main()
        elif target == 'Elbow-kmeans++':
            elbowKmeansPl.elbowKmeansPlGUI().Main()
        statusVar.set("Status : Completed Clustering")

    def judgePatternMiningAlg(self, target):
        self.root.destroy()
        if target == 'Frequent-spatial Pattern':
            periodicFrequentPattern.periodicFrequentPattern().Main()
        statusVar.set("Status : Completed Pattern Mining")

    def rootGUI(self):

        def getCategoricalType():
            if fillCategoricalVar.get() == "constant":
                constantValue_TB.grid(column=4, row=2, padx=20, pady=30)
            elif fillCategoricalVar.get() == "most_frequent":
                constantValue_TB.grid_remove()

        def getNumericType():
            if fillNumericVar.get() != 'constant':
                constantNumValue_TB.grid_remove()
            elif fillNumericVar.get() != 'mean' or 'median' or 'most_frequent':
                constantNumValue_TB.grid(column=4, row=2, padx=30, pady=30)

        def getfillType():
            if fillBothVar.get() == 'constant':
                constantValue_TB.grid(column=4, row=2, padx=20, pady=30)
            elif fillBothVar.get() == "most_frequent":
                constantValue_TB.grid_remove()

        def addOptions():
            if imputationType.get() == 'fill type - Categorical Values':
                fillNumeric_CB.grid_remove()
                fillBoth_CB.grid_remove()
                dropValue_CB.grid_remove()
                fillCategorical_CB.grid(column=3, row=2, padx=30, pady=30)
                fillCategorical_CB.bind('<<ComboboxSelected>>',
                                        lambda e: getCategoricalType())
                # print(imputationType.get())
            elif imputationType.get() == 'fill type - Numerical Values':
                fillCategorical_CB.grid_remove()
                fillBoth_CB.grid_remove()
                dropValue_CB.grid_remove()
                fillNumeric_CB.grid(column=3, row=2, padx=30, pady=30)
                fillNumeric_CB.bind('<<ComboboxSelected>>',
                                    lambda e: getNumericType())
                # print(imputationType.get())
            elif imputationType.get() == 'fill type - Both':
                fillCategorical_CB.grid_remove()
                fillNumeric_CB.grid_remove()
                dropValue_CB.grid_remove()
                fillBoth_CB.grid(column=3, row=2, padx=30, pady=30)
                fillBoth_CB.bind('<<ComboboxSelected>>',
                                 lambda e: getfillType())
                # print(imputationType.get())
            elif imputationType.get() == 'drop':
                fillCategorical_CB.grid_remove()
                fillNumeric_CB.grid_remove()
                fillBoth_CB.grid_remove()
                dropValue_CB.grid(column=3, row=2, padx=30, pady=30)
        def performImputation():
            df = pd.read_table(iFileNameHandlingNan.get(), sep='\t',index_col=0, encoding="shift-jis")
            df_index = df.index
            imputed_df = pd.DataFrame()
            if os.path.isfile(oFileNameHandlingNan.get() + '/processedData.tsv'):
                os.remove(oFileNameHandlingNan.get() + '/processedData.tsv')
            if imputationType.get() == 'fill type - Numerical Values':
                if fillNumericVar.get() == 'constant':
                    imputed_df = pd.DataFrame(constantImputer(df, fill_value=int(constantNumVal.get())))
                else:
                    imputed_df = pd.DataFrame(simpleImputer(df, strategy=str(fillNumericVar.get())))
            elif imputationType.get() == 'fill type - Categorical Values':
                if fillCategoricalVar.get() == 'constant':
                    imputed_df = pd.DataFrame(constantImputer(df, fill_value=str(constantVal.get())))
                else:
                    imputed_df = pd.DataFrame(simpleImputer(df, strategy=str(fillCategoricalVar.get())))
            elif imputationType.get() == 'drop':

                if dropVar.get() == 'all':
                    imputed_df = df.dropna(how='all').dropna(how='all', axis=1)
                elif dropVar.get() == 'all column':
                    imputed_df = df.dropna(how='all', axis=1)
                elif dropVar.get() == 'all row':
                    imputed_df = df.dropna(how='all')
                elif dropVar.get() == 'any':
                    imputed_df = df.dropna(how='any').dropna(how='any', axis=1)
                elif dropVar.get() == 'any column':
                    imputed_df = df.dropna(how='any')
                elif dropVar.get() == 'any row':
                    imputed_df = df.dropna(how='any', axis=1)
            imputed_df.index = df_index
            imputed_df.to_csv(oFileNameHandlingNan.get() + '/processedData.tsv', sep='\t', index=False,
                              float_format='%.2f')
            statusVar.set(str("Status : Imputation Completed !"))
            messagebox.showinfo('notification', 'Successfully completed')

        def mlImputation():
            df = pd.read_table(inputMLImputeFile.get(), sep='\t',index_col=0, encoding="shift-jis")
            df_index = df.index
            imputedML_df = pd.DataFrame()
            if os.path.isfile(outputMLImputeFile.get() + '/processedData.tsv'):
                os.remove(outputMLImputeFile.get() + '/processedData.tsv')
            if mlImputeType.get() == 'Linear Regression':
                imputedML_df = pd.DataFrame(knnImputer(df))
                print(mlImputeType.get())
            elif mlImputeType.get() == "KNN Imputation":
                imputedML_df = pd.DataFrame(knnImputer(df))
            imputedML_df.index = df_index
            imputedML_df.to_csv(outputMLImputeFile.get() + '/processedData.tsv', sep='\t', index=False,
                              float_format='%.2f')
            statusVar.set(str("Status : Imputation Completed !"))
            messagebox.showinfo('notification', 'Successfully completed')


        def imageFusionOptions(bin, scaleFactor):
            if bin.get():
                
                FWHMx1.set(math.floor(scaleFactor.get()/2))
                FWHMy1.set(math.floor(scaleFactor.get()/2))
                shiftx1.set(-scaleFactor.get())
                shifty1.set(-scaleFactor.get())
                rotationAngle1.set(35)
                FWHMx2.set(round(scaleFactor.get() * 2.5))
                FWHMy2.set(round(scaleFactor.get() * 2.5))
                shiftx2.set(scaleFactor.get())
                shifty2.set(scaleFactor.get())
                rotationAngle2.set(65)


        preprocessing = ['multi-band images', 'single-band temporal images']
        # imputation = ['Basic', 'Advanced']
        clusteringAlgorithms = {'Parameter tuning': ["Elbow-kmeans", "Elbow-kmeans++"],
                                'individual algorithm': ["k-Means/k-Means++", "fuzzy-kMeans", "DBScan",
                                                         "SpectralClustering", "MeanShift",
                                                         "OPTICS", "BIRCH", "AffinityPropagation"]}
        pamiAlgorithms = ['Periodic-frequent Pattern', 'Partial-periodic Pattern', 'Frequent-spatial Pattern',
                          'Periodic-frequent spatial Pattern']
        # classificationOptions = ['1folderValue', 'prediction']
        mineOptions = ['Temporal File', 'Neighborhood File', 'Mining']
        condition = ['<=', '>=', '<', '>']
        nanOpt = ['drop', 'fill type - Numerical Values', 'fill type - Categorical Values','fill type - Both']
        dropOpt = ['all', 'all column', 'all row', 'any', 'any column', 'any row']
        fillNumericOpt = ['mean','median','most_frequent','constant']
        fillCategoricalOpt = ['most_frequent','constant']
        fillBothOpt = ['most_frequent','constant']
        handlingNantype = ['statistical methods', 'machine learning methods']
        mlImputationOpt = ['Linear Regression','KNN Imputation']
        global oneNNEDFlag
        global oneNNDTWFlag
        global oneNNHausdorffFlag
        global oneNNmaxinormFlag
        global statusVar
        global iFileNameHandlingNan

        v2 = tk.StringVar()
        inputRasterFolderName = tk.StringVar()
        iFileNameHandlingNan = tk.StringVar()
        oFileNameHandlingNan = tk.StringVar()
        fileExtension = tk.StringVar()
        start_band_var = tk.StringVar()
        end_band_var = tk.StringVar()
        wranglingVar = tk.StringVar()
        imputationType = tk.StringVar()
        constantVal = tk.StringVar()
        dropVar = tk.StringVar()
        fillNumericVar = tk.StringVar()
        fillCategoricalVar = tk.StringVar()
        fillBothVar = tk.StringVar()
        constantNumVal = IntVar()
        convertVal = tk.StringVar()
        dropVar = tk.StringVar()
        strategyVar = tk.StringVar()
        fillNumericVar = tk.StringVar()
        fillCategoricalVar = tk.StringVar()
        fillBothVar = tk.StringVar()
        inputMLImputeFile = tk.StringVar()
        outputMLImputeFile = tk.StringVar()
        mlImputeType = tk.StringVar()
        inputTempFileName = tk.StringVar()
        inputTrainFile = tk.StringVar()
        inputTestFile = tk.StringVar()
        outputRecommendFolder = tk.StringVar()
        topkValue = tk.IntVar()
        conditionVar = tk.StringVar()
        thresholdVar = tk.StringVar()
        inputNeighborFileVar = tk.StringVar()
        outputNeighborFolderVar = tk.StringVar()
        pamiAlgVar = tk.StringVar()
        statusVar = tk.StringVar()
        statusVar.set('Status : Idle')
        outputFolderNameTab1 = tk.StringVar()
        oTempFolderVar = tk.StringVar()
        oneNNEDFlag = tk.BooleanVar()
        oneNNDTWFlag = tk.BooleanVar()
        oneNNHausdorffFlag = tk.BooleanVar()
        oneNNmaxinormFlag = tk.BooleanVar()
        fuzzyTSCFlag = tk.BooleanVar()
        fineImaget0InputFile = tk.StringVar()
        fineImaget1InputFile = tk.StringVar()
        coarseImaget0InputFile = tk.StringVar()
        coarseImaget1InputFile = tk.StringVar()
        imageFusionOutputFolderName = tk.StringVar()
        scaleFactor = IntVar()
        neighbours = IntVar()
        Iterations = IntVar()
        recommondValues = tk.BooleanVar()
        FWHMx1 = IntVar()
        FWHMy1 = IntVar()
        shiftx1 = IntVar()
        shifty1 = IntVar()
        rotationAngle1 = IntVar()
        FWHMx2 = IntVar()
        FWHMy2 = IntVar()
        shiftx2 = IntVar()
        shifty2 = IntVar()
        rotationAngle2 = IntVar()


        tabControl = ttk.Notebook(self.root)


        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tab3 = ttk.Frame(tabControl)
        tab4 = ttk.Frame(tabControl)
        tab5 = ttk.Frame(tabControl)
        tab6 = ttk.Frame(tabControl)
        tab7 = ttk.Frame(tabControl)

        subTab1 = ttk.Notebook(tab1, height=450, width=850)
        subTab1.pack(expand=True, side='top', fill='both')
        subTab2 = ttk.Notebook(tab2, height=450, width=850)
        subTab2.pack(expand=True, side='top', fill='both')
        subTab3 = ttk.Notebook(tab3, height=450, width=850)
        subTab3.pack(expand=True, side='top', fill='both')
        # subTab4 = ttk.Notebook(tab4, height=450, width=850)
        # subTab4.pack(expand=True, side='top', fill='both')
        subTab6 = ttk.Notebook(tab6, height=450, width=850)
        subTab6.pack(expand=True, side='top', fill='both')
        subTab7 = ttk.Notebook(tab7, height=450, width=850)
        subTab7.pack(expand=True, side='top', fill='both')

        tabControl.add(tab1, text='Preprocessing')
        tabControl.add(tab2, text='Imputation')
        tabControl.add(tab3, text='Clustering')
        tabControl.add(tab4, text='Image Fusion')
        tabControl.add(tab5, text='Recommender System')
        tabControl.add(tab6, text='Pattern Mining')
        tabControl.add(tab7, text='Prediction')
        tabControl.pack(expand=1, fill="both")

        variantID = tk.IntVar()
        # variantID2 = tk.IntVar()
        tk.Radiobutton(tab4, text="Improved HISTIF", padx = 20, variable=variantID, value=1).grid(column=0, row=1, padx=10, pady=10, sticky='W')
        tk.Radiobutton(tab4, text="Standard HSITIF", padx = 20, variable=variantID, value=2).grid(column=1, row=1, padx=10, pady=10, sticky='W')
        
        fineImaget0InputFile_label = ttk.Label(tab4, text='fine Image at t0')
        fineImaget0InputFile_label.grid(column=0, row=2, padx=10, pady=10, sticky='W')
        fineImaget0InputFile_TB = ttk.Entry(tab4, textvariable=fineImaget0InputFile, width=40)
        fineImaget0InputFile_TB.grid(column=1, row=2, padx=10, pady=10)
        fineImaget0InputFile_B = tk.Button(tab4, text='Browse',
                                    command=lambda: fineImaget0InputFile.set(str(self.uploadInputFile())))
        fineImaget0InputFile_B.grid(row=2, column=2, padx=10, pady=10)

        coarseImaget0InputFile_label = ttk.Label(tab4, text='coarse Image at t0')
        coarseImaget0InputFile_label.grid(column=0, row=3, padx=10, pady=10, sticky='W')
        coarseImaget0InputFile_TB = ttk.Entry(tab4, textvariable=coarseImaget0InputFile, width=40)
        coarseImaget0InputFile_TB.grid(column=1, row=3, padx=10, pady=10)
        coarseImaget0InputFile_B = tk.Button(tab4, text='Browse',
                                   command=lambda: coarseImaget0InputFile.set(str(self.uploadInputFile())))
        coarseImaget0InputFile_B.grid(row=3, column=2, padx=10, pady=10)

        coarseImaget1InputFile_label = ttk.Label(tab4, text='coarse Image at t1')
        coarseImaget1InputFile_label.grid(column=0, row=4, padx=10, pady=10, sticky='W')
        coarseImaget1InputFile_TB = ttk.Entry(tab4, textvariable=coarseImaget1InputFile, width=40)
        coarseImaget1InputFile_TB.grid(column=1, row=4, padx=10, pady=10)
        coarseImaget1InputFile_B = tk.Button(tab4, text='Browse',
                                             command=lambda: coarseImaget1InputFile.set(str(self.uploadInputFile())))
        coarseImaget1InputFile_B.grid(row=4, column=2, padx=10, pady=10)

        fineImaget1InputFile_label = ttk.Label(tab4, text='fine Image at t1 ( For Evaluation [Optional])')
        fineImaget1InputFile_label.grid(column=0, row=5, padx=10, pady=10, sticky='W')
        fineImaget1InputFile_TB = ttk.Entry(tab4, textvariable=fineImaget1InputFile, width=40)
        fineImaget1InputFile_TB.grid(column=1, row=5, padx=10, pady=10)
        fineImaget1InputFile_B = tk.Button(tab4, text='Browse',
                                           command=lambda: fineImaget1InputFile.set(str(self.uploadInputFile())))
        fineImaget1InputFile_B.grid(row=5, column=2, padx=10, pady=10)

        imageFusionOutputFolderName_label = ttk.Label(tab4, text='Select Output Folder ')
        imageFusionOutputFolderName_label.grid(column=0, row=6, padx=10, pady=10, sticky='W')
        imageFusionOutputFolderName_TB = ttk.Entry(tab4, textvariable=imageFusionOutputFolderName, width=40)
        imageFusionOutputFolderName_TB.grid(column=1, row=6, padx=10, pady=10)
        imageFusionOutputFolderName_B = tk.Button(tab4, text='Browse',
                                           command=lambda: imageFusionOutputFolderName.set(str(self.uploadOutputDir())))
        imageFusionOutputFolderName_B.grid(row=6, column=2, padx=10, pady=10)

        parameters_label = ttk.Label(tab4, text='Parameters')
        parameters_label.grid(column=1, row=7, padx=10, pady=10, sticky='W')

        scaleFactor_label = ttk.Label(tab4, text='Scale Factor')
        scaleFactor_label.grid(column=0, row=8, padx=10, pady=5, sticky='W')
        scaleFactor_TB = ttk.Entry(tab4, textvariable=scaleFactor,width =10)
        scaleFactor_TB.grid(column=1, row=8, padx=1, pady=5, sticky='W')


        iterations_label = ttk.Label(tab4, text='Iterations')
        iterations_label.grid(column=0, row=9, padx=10, pady=5, sticky='W')
        iterations_TB = ttk.Entry(tab4, textvariable=Iterations, width=10)
        iterations_TB.grid(column=1, row=9, padx=1, pady=5, sticky='W')

        Neighbours_label = ttk.Label(tab4, text='Neighbours')
        Neighbours_label.grid(column=0, row=10, padx=10, pady=5, sticky='W')
        Neighbours_TB = ttk.Entry(tab4, textvariable=neighbours, width=10)
        Neighbours_TB.grid(column=1, row=10, padx=1, pady=5, sticky='W')

        bin = BooleanVar()
        bin.set(True)

        recommondValues_CHB = ttk.Checkbutton(tab4, text=' Use Recommended Values ', variable=recommondValues,command= lambda :imageFusionOptions(bin, scaleFactor))
        recommondValues_CHB.grid(row=11, column=1, padx=10, pady=10)

        FWHMx_label1 = ttk.Label(tab4, text='FWHM_x')
        FWHMx_label1.grid(column=0, row=12, padx=10, pady=5, sticky='E')
        FWHMx_TB1 = ttk.Entry(tab4, textvariable=FWHMx1, width=10)
        FWHMx_TB1.grid(column=1, row=12, padx=1, pady=1, sticky='W')
        FWHMx_label2 = ttk.Label(tab4, text='to')
        FWHMx_label2.grid(column=2, row=12, padx=1, pady=1, sticky='W')
        FWHMx_TB2 = ttk.Entry(tab4, textvariable=FWHMx2, width=10)
        FWHMx_TB2.grid(column=3, row=12, padx=1, pady=1, sticky='W')

        FWHMy_label1 = ttk.Label(tab4, text='FWHM_y')
        FWHMy_label1.grid(column=0, row=13, padx=10, pady=5, sticky='E')
        FWHMy_TB1 = ttk.Entry(tab4, textvariable=FWHMy1, width=10)
        FWHMy_TB1.grid(column=1, row=13, padx=1, pady=5, sticky='W')
        FWHMy_label2 = ttk.Label(tab4, text='to')
        FWHMy_label2.grid(column=2, row=13, padx=1, pady=1, sticky='W')
        FWHMy_TB2 = ttk.Entry(tab4, textvariable=FWHMy2, width=10)
        FWHMy_TB2.grid(column=3, row=13, padx=1, pady=5, sticky='W')

        shiftx_label1 = ttk.Label(tab4, text='shift_x')
        shiftx_label1.grid(column=0, row=14, padx=10, pady=5, sticky='E')
        shiftx_TB1 = ttk.Entry(tab4, textvariable=shiftx1, width=10)
        shiftx_TB1.grid(column=1, row=14, padx=1, pady=5, sticky='W')
        shiftx_label2 = ttk.Label(tab4, text='to')
        shiftx_label2.grid(column=2, row=14, padx=1, pady=1, sticky='W')
        shiftx_TB2 = ttk.Entry(tab4, textvariable=shiftx2, width=10)
        shiftx_TB2.grid(column=3, row=14, padx=1, pady=5, sticky='W')

        shifty_label1 = ttk.Label(tab4, text='shift_y')
        shifty_label1.grid(column=0, row=15, padx=10, pady=5, sticky='E')
        shifty_TB1 = ttk.Entry(tab4, textvariable=shifty1, width=10)
        shifty_TB1.grid(column=1, row=15, padx=1, pady=5, sticky='W')
        shifty_label2 = ttk.Label(tab4, text='to')
        shifty_label2.grid(column=2, row=15, padx=1, pady=1, sticky='W')
        shifty_TB2 = ttk.Entry(tab4, textvariable=shifty2, width=10)
        shifty_TB2.grid(column=3, row=15, padx=1, pady=5, sticky='W')

        rotationAngle_label1 = ttk.Label(tab4, text='rotation angle')
        rotationAngle_label1.grid(column=0, row=16, padx=10, pady=5, sticky='E')
        rotationAngle_TB1 = ttk.Entry(tab4, textvariable=rotationAngle1, width=10)
        rotationAngle_TB1.grid(column=1, row=16, padx=1, pady=5, sticky='W')
        rotationAngle_label2 = ttk.Label(tab4, text='to')
        rotationAngle_label2.grid(column=2, row=16, padx=1, pady=1, sticky='W')
        rotationAngle_TB2 = ttk.Entry(tab4, textvariable=rotationAngle2, width=10)
        rotationAngle_TB2.grid(column=3, row=16, padx=1, pady=5, sticky='W')
        submit = tk.Button(tab4, text='submit', command = lambda : self.HISTIF(str(fineImaget0InputFile.get()), str(coarseImaget0InputFile.get()), str(coarseImaget1InputFile.get()), str(imageFusionOutputFolderName.get()), neighbours.get(), np.array([[FWHMx1.get(), FWHMx2.get()], [FWHMy1.get(), FWHMy2.get()], [shiftx1.get(), shiftx2.get()], [shifty1.get(), shifty2.get()], [rotationAngle1.get(), rotationAngle2.get()]]), Iterations.get(), variantID.get()))

        submit.grid(row=17, column=1,padx = 10,pady =20)
        submit.bind('<1>', lambda e: statusVar.set(str("Status : Running Image Fusion")))














                #--------------------------------------------------------------------------------------




        # Frame(subTab5)
        # subTab5.add(subFrame5, text=classificationOption)
        oneNNED_CHB = ttk.Checkbutton(tab5, text='1NNED', variable=oneNNEDFlag)
        oneNNED_CHB.grid(row=0, column=0, padx=30, pady=30)
        oneNNDTW_CHB = ttk.Checkbutton(tab5, text='1NNDTW', variable=oneNNDTWFlag)
        oneNNDTW_CHB.grid(row=0, column=1, padx=30, pady=30)
        oneNNHausdorff_CHB = ttk.Checkbutton(tab5, text='1NNHausdorff', variable=oneNNHausdorffFlag)
        oneNNHausdorff_CHB.grid(row=1, column=0, padx=30, pady=30)
        oneNNmaxinorm_CHB = ttk.Checkbutton(tab5, text='1NNmaxinorm', variable=oneNNmaxinormFlag)
        oneNNmaxinorm_CHB.grid(row=1, column=1, padx=30, pady=30)
        fuzzyTSC_CHB = ttk.Checkbutton(tab5, text='FuzzyTSC', variable=fuzzyTSCFlag)
        oneNNmaxinorm_CHB.grid(row=1, column=1, padx=30, pady=30)

        iTrainingFile_label = ttk.Label(tab5, text='input training file')
        iTrainingFile_label.grid(column=0, row=2, padx=60, pady=30, sticky='W')
        iTrainingFile_TB = ttk.Entry(tab5, textvariable=inputTrainFile, width=40)
        iTrainingFile_TB.grid(column=1, row=2, padx=60, pady=30)
        iTrainingFile_B = tk.Button(tab5, text='Browse',
                                    command=lambda: inputTrainFile.set(str(self.uploadInputFile())))
        iTrainingFile_B.grid(row=2, column=2, padx=60, pady=30)

        iTestingFile_label = ttk.Label(tab5, text='input test file')
        iTestingFile_label.grid(column=0, row=3, padx=60, pady=30, sticky='W')
        iTestingFile_TB = ttk.Entry(tab5, textvariable=inputTestFile, width=40)
        iTestingFile_TB.grid(column=1, row=3, padx=60, pady=30)
        iTestingFile_B = tk.Button(tab5, text='Browse',
                                   command=lambda: inputTestFile.set(str(self.uploadInputFile())))
        iTestingFile_B.grid(row=3, column=2, padx=60, pady=30)

        outputRecommendFolder_label = ttk.Label(tab5, text='output Folder')
        outputRecommendFolder_label.grid(column=0, row=4, padx=60, pady=30, sticky='W')
        outputRecommendFolder_TB = ttk.Entry(tab5, textvariable=outputRecommendFolder, width=40)
        outputRecommendFolder_TB.grid(column=1, row=4, padx=60, pady=30)
        outputRecommendFolder_B = tk.Button(tab5, text='Browse',
                                   command=lambda: outputRecommendFolder.set(str(self.uploadOutputDir())))
        outputRecommendFolder_B.grid(row=4, column=2, padx=60, pady=30)

        topkValue_label = ttk.Label(tab5, text='select top K value')
        topkValue_label.grid(column=0, row=5, padx=60, pady=30, sticky='W')
        topkValue_TB = ttk.Entry(tab5, textvariable=topkValue, width=40)
        topkValue_TB.grid(column=1, row=5, padx=60, pady=30)



        submit = tk.Button(tab5, text='submit',
                           command=lambda: self.recommender(str(inputTrainFile.get()), str(inputTestFile.get()),
                                                           str(outputRecommendFolder.get()),topkValue.get()))

        submit.grid(row=6, column=0)
        submit.bind('<1>', lambda e: statusVar.set(str("Status : Running Recommender System")))

        for option in preprocessing:
            subFrame1 = ttk.Frame(subTab1)
            subTab1.add(subFrame1, text=option)

            label1 = ttk.Label(subFrame1, text='Select the folder containing raster files:')
            label2 = ttk.Label(subFrame1, text='Enter the file extension of the raster files:')
            label3 = ttk.Label(subFrame1, text='Select output folder:')
            label4 = ttk.Label(subFrame1, text='Initial band number')
            label5 = ttk.Label(subFrame1, text='Final band number')
            label1.grid(column=0, row=0, padx=30, pady=30, sticky='W')
            label2.grid(column=0, row=1, padx=30, pady=30, sticky='W')
            label3.grid(column=0, row=2, padx=30, pady=30, sticky='W')
            label4.grid(column=0, row=3, padx=30, pady=30, sticky='W')
            label5.grid(column=0, row=4, padx=30, pady=30, sticky='W')

            e1 = tk.Entry(subFrame1, textvariable=inputRasterFolderName, width=40)
            e2 = tk.Entry(subFrame1, textvariable=fileExtension, width=40)
            e3 = tk.Entry(subFrame1, textvariable=outputFolderNameTab1, width=40)

            e1.grid(row=0, column=1)
            e2.grid(row=1, column=1)
            e3.grid(row=2, column=1)

            button1 = tk.Button(subFrame1, text='Browse',
                                command=lambda: inputRasterFolderName.set(str(self.uploadInputDir())))
            button1.grid(row=0, column=2, padx=30)
            button2 = tk.Button(subFrame1, text='Browse',
                                command=lambda: outputFolderNameTab1.set(str(self.uploadOutputDir())))
            button2.grid(row=2, column=2, padx=30)
            a = outputFolderNameTab1.get()
            start_band_TB = tk.Entry(subFrame1, textvariable=start_band_var, width=5)
            start_band_TB.grid(row=3, column=1)
            end_band_TB = tk.Entry(subFrame1, textvariable=end_band_var, width=5)
            end_band_TB.grid(row=4, column=1)

            if option == 'multi-band images':
                label4.grid()
                label5.grid()
                start_band_TB.grid()
                end_band_TB.grid()
                submit = tk.Button(subFrame1, text='submit',
                                   command=lambda: self.rasterToHorizontal(str(inputRasterFolderName.get()),
                                                                           str(fileExtension.get()),
                                                                           str(outputFolderNameTab1.get()),
                                                                           int(start_band_var.get()),
                                                                           int(end_band_var.get())))
                submit.bind('<1>', lambda e: statusVar.set(str("Status : Running Preprocessing")))

                submit.grid(row=6, column=0)
            elif option == 'single-band temporal images':
                label4.grid_remove()
                label5.grid_remove()
                start_band_TB.grid_remove()
                end_band_TB.grid_remove()
                submit = tk.Button(subFrame1, text='submit',
                                   command=lambda: self.rasterToVertical(str(inputRasterFolderName.get()),
                                                                         str(fileExtension.get()),
                                                                         str(outputFolderNameTab1.get())))
                submit.bind('<1>', lambda e: statusVar.set(str("Status : Running Preprocessing")))
                submit.grid(row=6, column=0)

        for option in handlingNantype:
            subFrame2 = ttk.Frame(subTab2)
            subTab2.add(subFrame2, text=option)
            if option == "statistical methods":
                constantValue_TB = ttk.Entry(subFrame2, textvariable=constantVal, width=10)
                constantNumValue_TB = ttk.Entry(subFrame2, textvariable=constantNumVal, width=10)
                dropValue_CB = ttk.Combobox(subFrame2, textvariable=dropVar, values=dropOpt, state='readonly')
                fillNumeric_CB = ttk.Combobox(subFrame2, textvariable=fillNumericVar, values=fillNumericOpt, state='readonly')
                fillCategorical_CB = ttk.Combobox(subFrame2, textvariable=fillCategoricalVar, values=fillCategoricalOpt, state='readonly')
                fillBoth_CB = ttk.Combobox(subFrame2, textvariable=fillBothVar, values=fillBothOpt, state='readonly')
                label6 = ttk.Label(subFrame2, text='Select the inputFile')
                label6.grid(row=0, column=0, padx=30, pady=30)
                label7 = ttk.Label(subFrame2, text='Select the outputFolder')
                label7.grid(row=1, column=0, padx=30, pady=30)
                label7 = ttk.Label(subFrame2, text='Select the handling type')
                label7.grid(row=2, column=0, padx=30, pady=30)
                iFileName_TB = tk.Entry(subFrame2, textvariable=iFileNameHandlingNan, width=40)
                iFileName_TB.grid(row=0, column=1, padx=30, pady=30)
                iFileName_B = tk.Button(subFrame2, text='Browse',
                                        command=lambda: iFileNameHandlingNan.set(str(self.uploadInputFile())))
                oFileName_TB = tk.Entry(subFrame2, textvariable=oFileNameHandlingNan, width=40)
                oFileName_TB.grid(row=1, column=1, padx=30, pady=30)
                oFIleName_B = tk.Button(subFrame2, text='Browse',
                                        command=lambda: oFileNameHandlingNan.set(str(self.uploadOutputDir())))
                nanValueTreat_CB = ttk.Combobox(subFrame2, textvariable=imputationType, value=nanOpt, state='readonly',width=30)
                nanValueTreat_CB.grid(row=2, column=1, padx=30, pady=30)
                # nanValueTreat_CB.set(nanOpt[0])
                nanValueTreat_CB.bind('<<ComboboxSelected>>',
                                      lambda e: addOptions())

                iFileName_B.grid(row=0, column=2, padx=30)
                oFIleName_B.grid(row=1, column=2, padx=30)

                submit = tk.Button(subFrame2, text='submit', command=lambda: performImputation())
                submit.grid(row=3, column=0)
                submit.bind('<1>',
                            lambda e: statusVar.set("Status : Running Imputation"))
            elif option == "machine learning methods":
                mlMethod_CB = ttk.Combobox(subFrame2,textvariable=mlImputeType, values= mlImputationOpt, state='readonly')
                label8 = ttk.Label(subFrame2, text='Select the inputFile')
                label8.grid(row=0, column=0, padx=30, pady=30)
                label9 = ttk.Label(subFrame2, text='Select the outputFolder')
                label9.grid(row=1, column=0, padx=30, pady=30)
                label10 = ttk.Label(subFrame2, text='Select the handling type')
                label10.grid(row=2, column=0, padx=30, pady=30)
                iFileNameML_TB = tk.Entry(subFrame2, textvariable=inputMLImputeFile, width=40)
                iFileNameML_TB.grid(row=0, column=1, padx=30, pady=30)
                iFileNameML_B = tk.Button(subFrame2, text='Browse',
                                        command=lambda: inputMLImputeFile.set(str(self.uploadInputFile())))
                oFileNameML_TB = tk.Entry(subFrame2, textvariable=outputMLImputeFile, width=40)
                oFileNameML_TB.grid(row=1, column=1, padx=30, pady=30)
                oFIleNameML_B = tk.Button(subFrame2, text='Browse',
                                        command=lambda: outputMLImputeFile.set(str(self.uploadOutputDir())))
                nanValueTreatML_CB = ttk.Combobox(subFrame2, textvariable=mlImputeType, value=mlImputationOpt, state='readonly',
                                                width=30)
                nanValueTreatML_CB.grid(row=2, column=1, padx=30, pady=30)
                # nanValueTreat_CB.set(nanOpt[0])
                nanValueTreatML_CB.bind('<<ComboboxSelected>>',
                                      lambda e: addOptions())

                iFileNameML_B.grid(row=0, column=2, padx=30)
                oFIleNameML_B.grid(row=1, column=2, padx=30)

                submit = tk.Button(subFrame2, text='submit', command=lambda: mlImputation())
                submit.grid(row=3, column=0)
                submit.bind('<1>',
                            lambda e: statusVar.set("Status : Running Imputation"))

        for algorithm in clusteringAlgorithms.keys():
            subFrame3 = ttk.Frame(subTab3)
            subTab3.add(subFrame3, text=algorithm)
            cb2 = ttk.Combobox(subFrame3, textvariable=v2, state='readonly')
            cb2.place(relx=0.25, rely=0.5, relwidth=0.5)
            if algorithm == 'Parameter tuning':
                cb2.config(values=clusteringAlgorithms['Parameter tuning'])
            elif algorithm == 'individual algorithm':
                cb2.config(values=clusteringAlgorithms['individual algorithm'])
            submit = ttk.Button(subFrame3, text='submit', command=lambda: self.judgeClusteringAlg(v2.get()))
            submit.place(relx=0.378, rely=0.7, relwidth=0.25, relheight=0.125)
            submit.bind('<1>',
                        lambda e: statusVar.set("Status : Running Clustering"))


        for mineOption in mineOptions:
            subFrame6 = ttk.Frame(subTab6)
            subTab6.add(subFrame6, text=mineOption)
            if mineOption == 'Temporal File':
                iTempFile_label = ttk.Label(subFrame6, text='input file')
                iTempFile_label.grid(column=0, row=0, padx=60, pady=30, sticky='W')
                iTempFile_TB = ttk.Entry(subFrame6, textvariable=inputTempFileName, width=40)
                iTempFile_TB.grid(column=1, row=0, padx=60, pady=30)
                iTempFile_B = tk.Button(subFrame6, text='Browse',
                                        command=lambda: inputTempFileName.set(str(self.uploadInputFile())))
                iTempFile_B.grid(row=0, column=2, padx=60, pady=30)

                oTempFolder_label = ttk.Label(subFrame6, text='output folder')
                oTempFolder_label.grid(column=0, row=1, padx=60, pady=30, sticky='W')
                oTempFolder_TB = ttk.Entry(subFrame6, textvariable=oTempFolderVar, width=40)
                oTempFolder_TB.grid(column=1, row=1, padx=60, pady=30)
                oTempFolder_B = tk.Button(subFrame6, text='Browse',
                                          command=lambda: oTempFolderVar.set(str(self.uploadOutputDir())))
                oTempFolder_B.grid(row=1, column=2, padx=60, pady=30)

                condition_label = ttk.Label(subFrame6, text='condition')
                condition_label.grid(column=0, row=2, padx=60, pady=30, sticky='W')
                condition_CB = ttk.Combobox(subFrame6, textvariable=conditionVar, values=condition, state='readonly')
                condition_CB.grid(column=1, row=2, padx=60, pady=30)

                threshold_label = ttk.Label(subFrame6, text='threshold')
                threshold_label.grid(column=0, row=3, padx=60, pady=30)
                threshold_TB = ttk.Entry(subFrame6, textvariable=thresholdVar)
                threshold_TB.grid(column=1, row=3, padx=60, pady=30)

                submit = tk.Button(subFrame6, text='submit',
                                   command=lambda: self.patternMiningTempFile(inputTempFileName.get(),
                                                                              oTempFolderVar.get()
                                                                              , conditionVar.get(),
                                                                              int(thresholdVar.get())))
                submit.grid(row=4, column=0, pady=30)
                submit.bind('<1>',
                            lambda e: statusVar.set("Status : Running Pattern Mining"))

            elif mineOption == 'Mining':
                patternMiningAlg_label = ttk.Label(subFrame6, text='select the algorithm')
                patternMiningAlg_label.grid(column=0, row=0, padx=60, pady=30, sticky='W')

                patternMiningAlg_CB = ttk.Combobox(subFrame6, textvariable=pamiAlgVar, values=pamiAlgorithms,
                                                   state='readonly', width=50)
                patternMiningAlg_CB.grid(column=1, row=0, padx=60, pady=30)

                submit = tk.Button(subFrame6, text='submit',
                                   command=lambda: self.judgePatternMiningAlg(pamiAlgVar.get()))
                submit.grid(row=1, column=0, pady=30)
                submit.bind('<1>',
                            lambda e: statusVar.set("Status : Running Pattern Mining"))

            elif mineOption == 'Neighborhood File':
                iNeighborFile_label = ttk.Label(subFrame6, text='Select the file:')
                iNeighborFile_label.grid(row=0, column=0, padx=60, pady=30, sticky='W')
                iNeighborFile_TB = tk.Entry(subFrame6, textvariable=inputNeighborFileVar, width=40)
                iNeighborFile_TB.grid(row=0, column=1)

                iNeighborFile_B = tk.Button(subFrame6, text='Browse',
                                            command=lambda: inputNeighborFileVar.set(str(self.uploadInputFile())))
                iNeighborFile_B.grid(row=0, column=2, padx=60)

                outputNeighborFolder_label = ttk.Label(subFrame6, text='Select output folder:')
                outputNeighborFolder_label.grid(column=0, row=1, padx=60, pady=30, sticky='W')
                outputNeighborFolder_TB = tk.Entry(subFrame6, textvariable=outputNeighborFolderVar, width=40)
                outputNeighborFolder_TB.grid(row=1, column=1)

                outputNeighborFolder_B = tk.Button(subFrame6, text='Browse',
                                                   command=lambda: outputNeighborFolderVar.set(
                                                       str(self.uploadOutputDir())))
                outputNeighborFolder_B.grid(row=1, column=2, padx=60)

                threshold_label = ttk.Label(subFrame6, text='threshold')
                threshold_label.grid(column=0, row=2, padx=60, pady=30)
                threshold_TB = ttk.Entry(subFrame6, textvariable=thresholdVar)
                threshold_TB.grid(column=1, row=2, padx=60, pady=30)

                submit = tk.Button(subFrame6, text='submit',
                                   command=lambda: self.patternMiningNeighbourFile(inputNeighborFileVar.get(),
                                                                                   outputNeighborFolderVar.get(),
                                                                                   int(thresholdVar.get())))
                submit.grid(row=3, column=0, pady=30)
                submit.bind('<1>',
                            lambda e: statusVar.set("Status : Running Pattern Mining"))

        status = Label(self.root, textvariable=statusVar, bd=1, relief="flat", anchor=W)
        status.pack(side=BOTTOM, padx=10, pady=5, fill='both')

        self.root.mainloop()


if __name__ == '__main__':
    GUImain().rootGUI()
