import Image 
import numpy as np
def ImageFeatures(Img):
    Whiten = np.loadtxt('Whiten.txt')
    Codes = np.loadtxt('Codes.txt')
    Mean = np.loadtxt('Mean.txt')
    FlatenImage = Image.flaten()
    CenterImage = Image - sum(FlatenImage)/len(FlatenImage)
    STD = np.var(FlatenImage)
    Normalised = CenterImage/ np.sqrt(float(STD + 10))
    # Applying Filters
    Normalised = np.dot((Normalised - Mean),Whiten)
    features = Normalised * D.codes
    return features

