package com.m.analiseTendenciaTransacao;

import lombok.Data;

import java.time.LocalDate;

@Data
public class Transaction {
    private int establishmentCode;
    private double amount;
    private LocalDate transactionDate;

    public Transaction() {}

    public Transaction(int establishmentCode, double amount, LocalDate transactionDate) {
        this.establishmentCode = establishmentCode;
        this.amount = amount;
        this.transactionDate = transactionDate;
    }

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
}
