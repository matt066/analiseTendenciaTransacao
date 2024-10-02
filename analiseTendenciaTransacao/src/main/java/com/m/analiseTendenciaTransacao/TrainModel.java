package com.m.analiseTendenciaTransacao;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class TrainModel {

    public static void main(String[] args) {
        // Configuração inicial do modelo
        int numInputs = 3;  // Define o número de entradas esperadas no modelo.
        int numOutputs = 2;  // Define o número de saídas, por exemplo, para uma classificação binária.
        int numHiddenNodes = 20;  // Define o número de nós na camada oculta.
        int batchSize = 10;  // Tamanho do lote para processamento durante o treinamento.

        // Configuração da rede neural
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.01))  // Configura o otimizador Adam com taxa de aprendizado de 0.01.
                .list()  // Início da configuração das camadas.
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)  // Função de ativação ReLU para a camada oculta.
                        .weightInit(WeightInit.XAVIER)  // Inicialização de pesos usando Xavier.
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)  // Função de ativação Softmax para a saída.
                        .nIn(numHiddenNodes).nOut(numOutputs)
                        .build())
                .build();

        // Criação e inicialização do modelo
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Geração de dados fictícios para treinamento
        List<DataSet> allData = generateDummyData(100000, numInputs, numOutputs);
        DataSet fullDataSet = DataSet.merge(allData);

        // Divisão dos dados em conjuntos de treino e teste
        SplitTestAndTrain split = fullDataSet.splitTestAndTrain(0.8); // 80% dos dados para treino e 20% para teste.

        // Treinamento do modelo
        DataSet trainingData = split.getTrain();
        DataSet testData = split.getTest();
        model.fit(new ListDataSetIterator<>(trainingData.asList(), batchSize));

        // Avaliação do modelo treinado
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);  // Criação do objeto de avaliação.
        INDArray output = model.output(testData.getFeatures());  // Execução do modelo no conjunto de teste.
        eval.eval(testData.getLabels(), output);  // Avaliação do modelo baseado nas saídas previstas e reais.
        System.out.println(eval.stats());

        // Salvamento do modelo treinado
        File modelFile = new File("model.zip");  // Localização do arquivo onde o modelo será salvo.
        try {
            ModelSerializer.writeModel(model, modelFile, true);
            System.out.println("Modelo salvo em: " + modelFile.getPath());
        } catch (Exception e) {
            System.out.println("Erro ao salvar o modelo: " + e.getMessage());
        }
    }

    // Método para gerar dados fictícios para o treinamento
    private static List<DataSet> generateDummyData(int numExamples, int numInputs, int numOutputs) {
        List<DataSet> dataSets = new ArrayList<>();
        for (int i = 0; i < numExamples; i++) {
            double[] features = new double[numInputs];
            double[] labels = new double[numOutputs];
            for (int j = 0; j < numInputs; j++) {
                features[j] = Math.random();  // Geração de características fictícias.
            }
            labels[0] = features[0] > 0.5 ? 1 : 0;  // Classificação fictícia baseada em uma regra simples.
            labels[1] = 1 - labels[0];
            INDArray featuresArray = Nd4j.create(features, new int[]{1, numInputs});
            INDArray labelsArray = Nd4j.create(labels, new int[]{1, numOutputs});
            dataSets.add(new DataSet(featuresArray, labelsArray));
        }
        return dataSets;
    }
}
