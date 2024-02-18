import os
import shutil


def main():
    root = os.getcwd() + "/saved/"
    for dm_dir in os.listdir(root):
        subroot1 = root + dm_dir + "/"

        # If dm directory is not empty go deeper
        if len(os.listdir(subroot1)) != 0:
            for ablation_dir in os.listdir(subroot1):
                subroot2 = subroot1 + ablation_dir + "/"

                # If ablation dir is not empty go deeper
                if len(os.listdir(subroot2)) != 0:
                    for model_dir in os.listdir(subroot2):
                        subroot3 = subroot2 + model_dir + "/"

                        # If model dir is not empty go deeper
                        if len(os.listdir(subroot3)) != 0:
                            for run_dir in os.listdir(subroot3):
                                subroot4 = subroot3 + run_dir + "/"
                                
                                # Check if checkpoints does not exist, and if it does, check if it has 0 files
                                if not(os.path.isdir(subroot4 + "checkpoints/")) or len(os.listdir(subroot4 + "checkpoints/"))  == 0:
                                    print("Now deleting " + subroot4)
                                    shutil.rmtree(subroot4)

                        # If now empty, delete this directory
                        if len(os.listdir(subroot3)) == 0:
                            print("Now deleting " + subroot3)
                            shutil.rmtree(subroot3)
                            
                # If now empty, delete this directory
                if len(os.listdir(subroot2)) == 0:
                    print("Now deleting " + subroot2)
                    shutil.rmtree(subroot2)
        
        # If now empty, delete this directory
        if len(os.listdir(subroot1)) == 0:
            print("Now deleting " + subroot1)
            shutil.rmtree(subroot1)

if __name__ == "__main__":
    main()