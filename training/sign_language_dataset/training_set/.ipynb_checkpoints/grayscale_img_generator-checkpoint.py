{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import cv2\n",
    "import os\n",
    "\n",
    "##############################\n",
    "# here we collect all the paths\n",
    "# to our train / test images\n",
    "pathToImages = []\n",
    "for root, dirs, files in os.walk(\".\", topdown=False): \n",
    "    for name in files:\n",
    "        path = os.path.join(root, name)\n",
    "        if path.endswith(\"jpg\"):\n",
    "            pathToImages.append(path) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "667\n"
     ]
    }
   ],
   "source": [
    "print(len(pathToImages))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loops through imagepaths to load images and labels into arrays\n",
    "for path in pathToImages:\n",
    "    # Reads image and returns np.array\n",
    "    img = cv2.imread(path) \n",
    "    img = cv2.resize(img, (128, 128)) \n",
    "    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "    cv2.imwrite(path, img)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
