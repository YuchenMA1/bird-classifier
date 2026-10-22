DATA_ROOT = "data"
TRAIN_IMAGE_DIR = "data/Train"      
TEST_IMAGE_DIR = "data/Test"       
TRAIN_TXT = "data/train.txt"     
TEST_TXT = "data/test.txt"        

OUTPUT_DIR = "outputs"
BEST_MODEL_NAME = "best_model.pth"  

SEED = 42                 
IMAGE_SIZE = 224         
BATCH_SIZE = 32           
EPOCHS = 20               
LEARNING_RATE_HEAD = 1e-3  
LEARNING_RATE_BACKBONE = 1e-4  
WEIGHT_DECAY = 1e-4       
LABEL_SMOOTHING = 0.1    
VAL_SPLIT = 0.1            
EARLY_STOP_PATIENCE = 5    


MODEL_NAME = "resnet50"
NUM_CLASSES = 200         
