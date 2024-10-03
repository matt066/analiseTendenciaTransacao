package com.m.analiseTendenciaTransacao;

import java.time.LocalDate;

public class Transaction {
    private int establishmentCode;
    private double amount;
    private LocalDate transactionDate;
    private double mediaAmount;  // Atributo para média do valor das transações
    private double taxaFraude;   // Atributo para taxa de fraude

    // Construtor
    public Transaction(int establishmentCode, double amount, LocalDate transactionDate, double mediaAmount, double taxaFraude) {
        this.establishmentCode = establishmentCode;
        this.amount = amount;
        this.transactionDate = transactionDate;
        this.mediaAmount = mediaAmount;
        this.taxaFraude = taxaFraude;
    }

    // Getters e Setters
    public int getEstablishmentCode() {
        return establishmentCode;
    }

    public void setEstablishmentCode(int establishmentCode) {
        this.establishmentCode = establishmentCode;
    }

    public double getAmount() {
        return amount;
    }

    public void setAmount(double amount) {
        this.amount = amount;
    }

    public LocalDate getTransactionDate() {
        return transactionDate;
    }

    public void setTransactionDate(LocalDate transactionDate) {
        this.transactionDate = transactionDate;
    }

    public double getMediaAmount() {
        return mediaAmount;
    }

    public void setMediaAmount(double mediaAmount) {
        this.mediaAmount = mediaAmount;
    }

    public double getTaxaFraude() {
        return taxaFraude;
    }

    public void setTaxaFraude(double taxaFraude) {
        this.taxaFraude = taxaFraude;
    }

    // Método para obter o valor médio de transações para o estabelecimento
    public double getAverageTransactionValue() {
        return this.mediaAmount;
    }

    // Método para definir o valor médio de transações para o estabelecimento
    public void setAverageTransactionValue(double averageValue) {
        this.mediaAmount = averageValue;
    }
}
