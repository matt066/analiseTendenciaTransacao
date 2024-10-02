package com.m.analiseTendenciaTransacao;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class LoadAndUseModel {

    public static void main(String[] args) {
        try {
            // Carrega o modelo a partir de um arquivo
            File modelFile = new File("model.zip");
            if (!modelFile.exists()) {
                System.out.println("Arquivo de modelo não encontrado: " + modelFile.getPath());
                return;
            }
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            System.out.println("Modelo carregado com sucesso!");

            // Imprime um resumo do modelo
            System.out.println("Sumário do Modelo:");
            System.out.println(model.summary());

            // Realiza uma inferência com dados de exemplo
            int numberOfInputFeatures = model.layerInputSize(0);  // Obtem o número de entradas esperadas pela primeira camada
            INDArray exampleInput = Nd4j.rand(1, numberOfInputFeatures);  // Gera dados aleatórios de entrada
            INDArray output = model.output(exampleInput);
            System.out.println("Saída da Rede Neural:");
            System.out.println(output);

        } catch (Exception e) {
            System.err.println("Erro ao carregar ou utilizar o modelo: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
