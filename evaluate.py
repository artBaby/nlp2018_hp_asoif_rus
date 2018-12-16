from create_questions import create_questions
from check_frequencies import check_freq
from create_models import create_models
from analogies_evaluation import analogies_evaluation
from doesnt_match_evaluation import doesnt_match_evaluation


print("\nCreating questions files...")
#create_questions()
print('Questions created in datasets directory\n')
print("Creating Frequencies for Harry Potter...")
check_freq("hp", "./books/hp_result.txt")
print("Frequencies for Harry Potter created in datasets directory\n")
print("Creating Frequencies for A Song of Ice and Fire ...")
check_freq("soiaf", "./books/asoif_result.txt")
print("Frequencies for A Song of Ice and Fire created in datasets directory\n")


# create_models("hp")
# create_models("asoif")


analogies_evaluation("hp")
analogies_evaluation("asoif")

doesnt_match_evaluation("hp", True)
doesnt_match_evaluation("asoif", True)