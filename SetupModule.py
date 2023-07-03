import os
import shutil

_separate_value_small = 500
_separate_value_medium = 750
_separate_value_large = 1000


# Directory Names
packOneDirectoryName = 'cane'
packTwoDirectoryName = 'cavallo'
packThreeDirectoryName = 'elefante'
packFourDirectoryName = 'faralla'
packFiveDirectoryName = 'gallina'
packSixDirectoryName = 'gatto'
packSevenDirectoryName = 'mucca'
packEightDirectoryName = 'pecora'
packNineDirectoryName = 'rango'
packTenDirectoryName = 'scoiatallo'

packOneFileSchema = 'cane{}.jpeg'
packTwoFileSchema = 'cavalo{}.jpeg'
packThreeFileSchema = 'elefante{}.jpeg'
packFourFileSchema = 'farfalla{}.jpeg'
packFiveFileSchema = 'gallina{}.jpeg'
packSixFileSchema = 'gatto{}.jpeg'
packSevenFileSchema = 'mucca{}.jpeg'
packEightFileSchema = 'pecora{}.jpeg'
packNineFileSchema = 'ragno{}.jpeg'
packTenFileSchema = 'scoittalo{}.jpeg'

saveFileName = 'learningSave.{}.h5'

# Define path to datasets
modelsDirectory = 'C:/ml/models/'
originalDatasetDirectory = 'C:/ml/data/kaggle/original/train'
baseDirectory = 'C:/ml/data/kaggle/processed/cats_and_dogs_small'

# Main Directories
trainDirectory = os.path.join(baseDirectory, 'train')
validationDirectory = os.path.join(baseDirectory, 'validation')
testDirectory = os.path.join(baseDirectory, 'test')

# Train Packs
trainPackOneDirectory = os.path.join(trainDirectory, packOneDirectoryName)
trainPackTwoDirectory = os.path.join(trainDirectory, packTwoDirectoryName)
trainPackThreeDirectory = os.path.join(trainDirectory, packThreeDirectoryName)
trainPackFourDirectory = os.path.join(trainDirectory, packFourDirectoryName)
trainPackFiveDirectory = os.path.join(trainDirectory, packFiveDirectoryName)
trainPackSixDirectory = os.path.join(trainDirectory, packSixDirectoryName)
trainPackSevenDirectory = os.path.join(trainDirectory, packSevenDirectoryName)
trainPackEightDirectory = os.path.join(trainDirectory, packEightDirectoryName)
trainPackNineDirectory = os.path.join(trainDirectory, packNineDirectoryName)
trainPackTenDirectory = os.path.join(trainDirectory, packTenDirectoryName)

# Validation Packs
validationPackOneDirectory = os.path.join(validationDirectory, packOneDirectoryName)
validationPackTwoDirectory = os.path.join(validationDirectory, packTwoDirectoryName)
validationPackThreeDirectory = os.path.join(validationDirectory, packThreeDirectoryName)
validationPackFourDirectory = os.path.join(validationDirectory, packFourDirectoryName)
validationPackFiveDirectory = os.path.join(validationDirectory, packFiveDirectoryName)
validationPackSixDirectory = os.path.join(validationDirectory, packSixDirectoryName)
validationPackSevenDirectory = os.path.join(validationDirectory, packSevenDirectoryName)
validationPackEightDirectory = os.path.join(validationDirectory, packEightDirectoryName)
validationPackNineDirectory = os.path.join(validationDirectory, packNineDirectoryName)
validationPackTenDirectory = os.path.join(validationDirectory, packTenDirectoryName)

# Test Packs
testPackOneDirectory = os.path.join(testDirectory, packOneDirectoryName)
testPackTwoDirectory = os.path.join(testDirectory, packTwoDirectoryName)
testPackThreeDirectory = os.path.join(testDirectory, packThreeDirectoryName)
testPackFourDirectory = os.path.join(testDirectory, packFourDirectoryName)
testPackFiveDirectory = os.path.join(testDirectory, packFiveDirectoryName)
testPackSixDirectory = os.path.join(testDirectory, packSixDirectoryName)
testPackSevenDirectory = os.path.join(testDirectory, packSevenDirectoryName)
testPackEightDirectory = os.path.join(testDirectory, packEightDirectoryName)
testPackNineDirectory = os.path.join(testDirectory, packNineDirectoryName)
testPackTenDirectory = os.path.join(testDirectory, packTenDirectoryName)


def check_directories():
    if not os.path.exists(trainDirectory):
        os.mkdir(trainDirectory)

    if not os.path.exists(validationDirectory):
        os.mkdir(validationDirectory)

    if not os.path.exists(testDirectory):
        os.mkdir(testDirectory)

    # Train
    if not os.path.exists(trainPackOneDirectory):
        os.mkdir(trainPackOneDirectory)

    if not os.path.exists(trainPackTwoDirectory):
        os.mkdir(trainPackTwoDirectory)

    if not os.path.exists(trainPackThreeDirectory):
        os.mkdir(trainPackThreeDirectory)

    if not os.path.exists(trainPackFourDirectory):
        os.mkdir(trainPackFourDirectory)

    if not os.path.exists(trainPackFiveDirectory):
        os.mkdir(trainPackFiveDirectory)

    if not os.path.exists(trainPackSixDirectory):
        os.mkdir(trainPackSixDirectory)

    if not os.path.exists(trainPackSevenDirectory):
        os.mkdir(trainPackSevenDirectory)

    if not os.path.exists(trainPackEightDirectory):
        os.mkdir(trainPackEightDirectory)

    if not os.path.exists(trainPackNineDirectory):
        os.mkdir(trainPackNineDirectory)

    if not os.path.exists(trainPackTenDirectory):
        os.mkdir(trainPackTenDirectory)

    # Validation
    if not os.path.exists(validationPackOneDirectory):
        os.mkdir(validationPackOneDirectory)

    if not os.path.exists(validationPackTwoDirectory):
        os.mkdir(validationPackTwoDirectory)

    if not os.path.exists(validationPackThreeDirectory):
        os.mkdir(validationPackThreeDirectory)

    if not os.path.exists(validationPackFourDirectory):
        os.mkdir(validationPackFourDirectory)

    if not os.path.exists(validationPackFiveDirectory):
        os.mkdir(validationPackFiveDirectory)

    if not os.path.exists(validationPackSixDirectory):
        os.mkdir(validationPackSixDirectory)

    if not os.path.exists(validationPackSevenDirectory):
        os.mkdir(validationPackSevenDirectory)

    if not os.path.exists(validationPackEightDirectory):
        os.mkdir(validationPackEightDirectory)

    if not os.path.exists(validationPackNineDirectory):
        os.mkdir(validationPackNineDirectory)

    if not os.path.exists(validationPackTenDirectory):
        os.mkdir(validationPackTenDirectory)

    # Test
    if not os.path.exists(testPackOneDirectory):
        os.mkdir(testPackOneDirectory)

    if not os.path.exists(testPackTwoDirectory):
        os.mkdir(testPackTwoDirectory)

    if not os.path.exists(testPackThreeDirectory):
        os.mkdir(testPackThreeDirectory)

    if not os.path.exists(testPackFourDirectory):
        os.mkdir(testPackFourDirectory)

    if not os.path.exists(testPackFiveDirectory):
        os.mkdir(testPackFiveDirectory)

    if not os.path.exists(testPackSixDirectory):
        os.mkdir(testPackSixDirectory)

    if not os.path.exists(testPackSevenDirectory):
        os.mkdir(testPackSevenDirectory)

    if not os.path.exists(testPackEightDirectory):
        os.mkdir(testPackEightDirectory)

    if not os.path.exists(testPackNineDirectory):
        os.mkdir(testPackNineDirectory)

    if not os.path.exists(testPackTenDirectory):
        os.mkdir(testPackTenDirectory)


def separate_files():
    # ========================================================================
    print('Separating Files of Pack One')
    file_names = [packOneFileSchema.format(i) for i in range(_separate_value_small)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(trainPackOneDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packOneFileSchema.format(i) for i in range(_separate_value_small, _separate_value_medium)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(validationPackOneDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packOneFileSchema.format(i) for i in range(_separate_value_medium, _separate_value_large)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(testPackOneDirectory, fileName)
        shutil.copy(source, destination)

    # ========================================================================
    print('Separating Files of Pack Two')
    file_names = [packTwoFileSchema.format(i) for i in range(_separate_value_small)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(trainPackTwoDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packTwoFileSchema.format(i) for i in range(_separate_value_small, _separate_value_medium)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(validationPackTwoDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packTwoFileSchema.format(i) for i in range(_separate_value_medium, _separate_value_large)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(testPackTwoDirectory, fileName)
        shutil.copy(source, destination)

    # ========================================================================
    print('Separating Files of Pack Three')
    file_names = [packThreeFileSchema.format(i) for i in range(_separate_value_small)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(trainPackThreeDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packThreeFileSchema.format(i) for i in range(_separate_value_small, _separate_value_medium)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(validationPackThreeDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packThreeFileSchema.format(i) for i in range(_separate_value_medium, _separate_value_large)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(testPackThreeDirectory, fileName)
        shutil.copy(source, destination)

    # ========================================================================
    print('Separating Files of Pack Four')
    file_names = [packFourFileSchema.format(i) for i in range(_separate_value_small)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(trainPackFourDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packFourFileSchema.format(i) for i in range(_separate_value_small, _separate_value_medium)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(validationPackFourDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packFourFileSchema.format(i) for i in range(_separate_value_medium, _separate_value_large)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(testPackFourDirectory, fileName)
        shutil.copy(source, destination)

    # ========================================================================
    print('Separating Files of Pack Five')
    file_names = [packFiveFileSchema.format(i) for i in range(_separate_value_small)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(trainPackFiveDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packFiveFileSchema.format(i) for i in range(_separate_value_small, _separate_value_medium)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(validationPackFiveDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packFiveFileSchema.format(i) for i in range(_separate_value_medium, _separate_value_large)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(testPackFiveDirectory, fileName)
        shutil.copy(source, destination)

    # ========================================================================
    print('Separating Files of Pack Six')
    file_names = [packSixFileSchema.format(i) for i in range(_separate_value_small)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(trainPackSixDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packSixFileSchema.format(i) for i in range(_separate_value_small, _separate_value_medium)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(validationPackSixDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packSixFileSchema.format(i) for i in range(_separate_value_medium, _separate_value_large)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(testPackSixDirectory, fileName)
        shutil.copy(source, destination)

    # ========================================================================
    print('Separating Files of Pack Seven')
    file_names = [packSevenFileSchema.format(i) for i in range(_separate_value_small)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(trainPackSevenDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packSevenFileSchema.format(i) for i in range(_separate_value_small, _separate_value_medium)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(validationPackSevenDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packSevenFileSchema.format(i) for i in range(_separate_value_medium, _separate_value_large)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(testPackSevenDirectory, fileName)
        shutil.copy(source, destination)

    # ========================================================================
    print('Separating Files of Pack Eight')
    file_names = [packEightFileSchema.format(i) for i in range(_separate_value_small)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(trainPackEightDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packEightFileSchema.format(i) for i in range(_separate_value_small, _separate_value_medium)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(validationPackEightDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packEightFileSchema.format(i) for i in range(_separate_value_medium, _separate_value_large)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(testPackEightDirectory, fileName)
        shutil.copy(source, destination)

    # ========================================================================
    print('Separating Files of Pack Nine')
    file_names = [packNineFileSchema.format(i) for i in range(_separate_value_small)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(trainPackNineDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packNineFileSchema.format(i) for i in range(_separate_value_small, _separate_value_medium)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(validationPackNineDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packNineFileSchema.format(i) for i in range(_separate_value_medium, _separate_value_large)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(testPackNineDirectory, fileName)
        shutil.copy(source, destination)

    # ========================================================================
    print('Separating Files of Pack Ten')
    file_names = [packTenFileSchema.format(i) for i in range(_separate_value_small)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(trainPackTenDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packTenFileSchema.format(i) for i in range(_separate_value_small, _separate_value_medium)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(validationPackTenDirectory, fileName)
        shutil.copy(source, destination)

    file_names = [packTenFileSchema.format(i) for i in range(_separate_value_medium, _separate_value_large)]
    for fileName in file_names:
        source = os.path.join(originalDatasetDirectory, fileName)
        destination = os.path.join(testPackTenDirectory, fileName)
        shutil.copy(source, destination)
