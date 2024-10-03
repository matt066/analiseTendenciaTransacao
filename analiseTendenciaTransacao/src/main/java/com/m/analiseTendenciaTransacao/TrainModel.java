package com.m.analiseTendenciaTransacao;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
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
        int numInputs = 4;  // Agora temos 4 entradas
        int numOutputs = 2;  // Saída binária (fraude/não fraude)
        int numHiddenNodes = 20;
        int batchSize = 10;

        // Configuração da rede neural
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.01))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs)
                        .build())
                .build();

        // Inicializando o modelo
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Gerando dados fictícios para treinamento
        List<DataSet> allData = generateDummyData(500000, numInputs, numOutputs);
        DataSet fullDataSet = DataSet.merge(allData);

        // Dividindo em treino e teste
        SplitTestAndTrain split = fullDataSet.splitTestAndTrain(0.8);
        DataSet trainingData = split.getTrain();
        DataSet testData = split.getTest();

        // Treinando o modelo
        model.fit(new ListDataSetIterator<>(trainingData.asList(), batchSize));

        // Avaliando o modelo
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());

        // Salvando o modelo treinado
        File modelFile = new File("model.zip");
        try {
            ModelSerializer.writeModel(model, modelFile, true);
            System.out.println("Modelo salvo em: " + modelFile.getPath());
        } catch (Exception e) {
            System.out.println("Erro ao salvar o modelo: " + e.getMessage());
        }
    }

    // Método para gerar dados fictícios com 4 entradas
    private static List<DataSet> generateDummyData(int numExamples, int numInputs, int numOutputs) {
        List<DataSet> dataSets = new ArrayList<>();
        for (int i = 0; i < numExamples; i++) {
            double[] features = new double[numInputs];
            double[] labels = new double[numOutputs];

            for (int j = 0; j < numInputs; j++) {
                features[j] = Math.random();  // Ajuste para dados fictícios
            }

            labels[0] = features[1] > 0.5 ? 1 : 0;  // Classificação baseada no valor da transação
            labels[1] = 1 - labels[0];

            INDArray featuresArray = Nd4j.create(features, new int[]{1, numInputs});
            INDArray labelsArray = Nd4j.create(labels, new int[]{1, numOutputs});
            dataSets.add(new DataSet(featuresArray, labelsArray));
        }
        return dataSets;
    }
}
