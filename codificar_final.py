#!/usr/bin/env python3
from Bio import SeqIO
import csv

# Diccionario de codificación one-hot
base_map = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

def encode_sequence(seq):
    """Convierte una secuencia en one-hot plano."""
    seq = seq.upper()
    vec = []
    for base in seq:
        vec.extend(base_map.get(base, [0, 0, 0, 0]))
    return vec


def get_max_len(fasta_files):
    """Primera pasada: encuentra la secuencia más larga entre todos los archivos."""
    max_len = 0
    for fasta, _label in fasta_files:
        for record in SeqIO.parse(fasta, "fasta"):
            seq_length = len(record.seq)
            if seq_length > max_len:
                max_len = seq_length
    return max_len


def process_fastas(fasta_files, output_csv, max_len):
    """Segunda pasada: codifica secuencia por secuencia con padding y escribe en CSV."""
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Ahora agregamos una columna para la clase
        writer.writerow(["ID", "label", "encoded"])

        for fasta, label in fasta_files:
            for record in SeqIO.parse(fasta, "fasta"):
                seq_vector = encode_sequence(str(record.seq))

                # Padding basado en la longitud máxima
                missing = (max_len - len(record.seq)) * 4
                if missing > 0:
                    seq_vector = seq_vector + [0] * missing

                writer.writerow([record.id, label, " ".join(map(str, seq_vector))])


if __name__ == "__main__":

    # Formato: (archivo, clase)
    fasta_files = [
        ("genes_var.fa", 1),   # clase 1
        ("genes_ref.fa", 0)    # clase 0
    ]

    output_csv = "secuencias_unificadas_onehot_clases.csv"

    print("➡ Primera pasada: buscando longitud máxima...")
    max_len = get_max_len(fasta_files)
    print(f"   Longitud máxima encontrada: {max_len} bases")

    print("➡ Segunda pasada: codificando secuencias...")
    process_fastas(fasta_files, output_csv, max_len)

    print("✔ Archivo generado:", output_csv)

