{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyM+h2mI8AsYlezPaAz3LZX7",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sofiaalbini/AI-UFFICIALE/blob/main/Dataset.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "from torch.utils.data import TensorDataset, DataLoader\n",
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
        "\n",
        "class Dataset:\n",
        "    def __init__(self, batch_size=16):\n",
        "        # Caricamento dataset Iris\n",
        "        iris = load_iris()\n",
        "        X, y = iris.data, iris.target\n",
        "\n",
        "        # Suddivisione in train/test con stratificazione\n",
        "        X_train, X_test, y_train, y_test = train_test_split(\n",
        "            X, y, test_size=0.2, random_state=42, stratify=y\n",
        "        )\n",
        "\n",
        "        # Standardizzazione delle feature\n",
        "        scaler = StandardScaler()\n",
        "        X_train = scaler.fit_transform(X_train)\n",
        "        X_test = scaler.transform(X_test)\n",
        "\n",
        "        # One-Hot Encoding delle etichette di training\n",
        "        ohe = OneHotEncoder(sparse_output=False)\n",
        "        y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))\n",
        "\n",
        "        # Conversione in tensori PyTorch\n",
        "        X_train_tensor = torch.from_numpy(X_train).float()\n",
        "        X_test_tensor = torch.from_numpy(X_test).float()\n",
        "        y_train_ohe_tensor = torch.from_numpy(y_train_ohe).float()\n",
        "        y_train_labels_tensor = torch.from_numpy(y_train).long()\n",
        "        y_test_labels_tensor = torch.from_numpy(y_test).long()\n",
        "\n",
        "        # Creazione dataset e dataloader\n",
        "        train_dataset = TensorDataset(X_train_tensor, y_train_ohe_tensor)\n",
        "        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
        "\n",
        "        # Salvataggio dati per eventuale uso\n",
        "        self.X_test = X_test_tensor\n",
        "        self.y_test_labels = y_test_labels_tensor\n",
        "        self.y_train_labels = y_train_labels_tensor  # Etichette originali (non one-hot)\n",
        "        self.y_train_ohe_tensor = y_train_ohe_tensor  # One-hot (tutto, non a batch)\n",
        "\n"
      ],
      "metadata": {
        "id": "hokFJ3Pp4UQj"
      },
      "execution_count": 1,
      "outputs": []
    }
  ]
}