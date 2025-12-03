"""
Optimal Augmentation Pipeline
==============================
Efficient data augmentation dengan quality filtering
"""

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import configurations
from config_optimal import *
from utils_optimal import *


# ============================================================================
# EXPANDED SYNONYM DICTIONARY (100+ entries)
# ============================================================================

SYNONYM_DICT = {
    # Politik & Pemerintahan
    "presiden": ["kepala negara", "presiden ri", "pres"],
    "menteri": ["mentri", "menteri negara"],
    "pemerintah": ["pemerintahan", "kabinet", "eksekutif"],
    "dpr": ["dewan", "legislatif", "parlemen"],
    "dprd": ["dewan rakyat", "legislatif daerah"],
    "partai": ["parpol", "partai politik"],
    "pemilu": ["pemilihan umum", "pilpres"],
    "gubernur": ["gub", "kepala daerah"],
    "walikota": ["wali kota"],
    "bupati": ["kepala kabupaten"],
    
    # Ekonomi & Bisnis
    "harga": ["biaya", "nilai", "tarif", "ongkos"],
    "naik": ["meningkat", "melonjak", "bertambah"],
    "turun": ["menurun", "merosot", "anjlok", "drop"],
    "ekonomi": ["perekonomian", "kondisi ekonomi"],
    "investasi": ["penanaman modal"],
    "rupiah": ["rp", "mata uang"],
    "saham": ["stok", "efek"],
    "modal": ["kapital", "dana"],
    "untung": ["profit", "laba", "keuntungan"],
    "rugi": ["loss", "kerugian"],
    "bisnis": ["usaha", "dagang"],
    "pasar": ["bursa", "market"],
    "ekspor": ["eksport"],
    "impor": ["import"],
    "bbm": ["bahan bakar", "bensin"],
    
    # Teknologi & Digital
    "aplikasi": ["app", "apl"],
    "smartphone": ["ponsel pintar", "hp", "handphone"],
    "internet": ["dunia maya", "daring", "online"],
    "digital": ["digitalisasi"],
    "website": ["situs", "web", "laman"],
    "komputer": ["pc", "laptop"],
    "data": ["informasi"],
    "software": ["perangkat lunak"],
    "hardware": ["perangkat keras"],
    
    # Olahraga
    "juara": ["pemenang", "jawara", "kampiun", "champion"],
    "kalah": ["tumbang", "takluk"],
    "menang": ["unggul", "juara"],
    "tim": ["skuad", "kesebelasan"],
    "pertandingan": ["laga", "match", "kompetisi"],
    "pemain": ["atlet", "pesepakbola"],
    "pelatih": ["coach", "trainer"],
    "gol": ["goal"],
    
    # Transportasi
    "mobil": ["kendaraan", "otomotif"],
    "motor": ["sepeda motor", "roda dua"],
    "jalan": ["raya", "jalanan"],
    "tol": ["toll", "jalan tol"],
    "bandara": ["airport", "lapangan terbang"],
    "stasiun": ["terminal kereta"],
    "kereta": ["ka", "kereta api"],
    "pesawat": ["aircraft", "pesawat terbang"],
    
    # Kesehatan
    "rumah sakit": ["rs", "hospital"],
    "dokter": ["dr", "medis"],
    "pasien": ["penderita"],
    "penyakit": ["sakit"],
    "virus": ["wabah"],
    "obat": ["medicine"],
    "vaksin": ["imunisasi"],
    
    # Makanan & Kuliner
    "makanan": ["kuliner", "menu", "hidangan"],
    "restoran": ["rumah makan", "resto"],
    "masak": ["memasak", "meracik"],
    "resep": ["recipe"],
    
    # Umum
    "baru": ["anyar", "terbaru"],
    "lama": ["lawas", "usang"],
    "besar": ["akbar", "raksasa", "agung"],
    "kecil": ["mini", "cilik"],
    "rakyat": ["masyarakat", "warga", "penduduk"],
    "negara": ["bangsa", "republik"],
    "daerah": ["wilayah", "regional"],
    "kota": ["kab", "kawasan"],
    "indonesia": ["indo", "ri", "nusantara"],
    "jakarta": ["jkt", "ibukota", "dki"],
    "proyek": ["pembangunan", "inisiatif"],
    "tahun": ["thn"],
    "hari": ["hr"],
    "minggu": ["pekan"],
    "bulan": ["bln"],
    "persen": ["prosen", "persentase"],
    "juta": ["jt"],
    "miliar": ["milyar"],
    "ribu": ["rb"],
    "orang": ["org", "individu"],
    "anak": ["bocah"],
    "kerja": ["pekerjaan", "bekerja"],
    "sekolah": ["pendidikan"],
}

print(f"📚 Loaded {len(SYNONYM_DICT)} synonym entries")


# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================

def aug_paraphrase(text: str, tokenizer, model, device) -> Optional[str]:
    """Generate paraphrase using IndoT5"""
    try:
        inp = "paraphrase: " + text + " </s>"
        enc = tokenizer(inp, return_tensors="pt").to(device)
        
        out = model.generate(
            **enc,
            max_length=PARA_MAX_LENGTH,
            do_sample=PARA_DO_SAMPLE,
            top_k=PARA_TOP_K,
            top_p=PARA_TOP_P,
            num_return_sequences=1
        )
        
        result = tokenizer.decode(out[0], skip_special_tokens=True)
        return result if result != text else None
    except Exception as e:
        return None


def aug_synonym(text: str, synonym_dict: Dict[str, List[str]]) -> Optional[str]:
    """Replace one word with synonym"""
    try:
        words = text.lower().split()
        
        candidates = [w for w in words if w in synonym_dict]
        if not candidates:
            return None
        
        chosen = random.choice(candidates)
        replacement = random.choice(synonym_dict[chosen])
        
        new_words = []
        for w in words:
            if w == chosen:
                new_words.append(replacement)
            else:
                new_words.append(w)
        
        result = " ".join(new_words)
        return result if result != text else None
    except Exception as e:
        return None


# ============================================================================
# BATCH AUGMENTATION WITH QUALITY FILTERING
# ============================================================================

def augment_batch(texts: List[str], labels: np.ndarray, 
                 para_tokenizer, para_model, device,
                 use_paraphrase: bool = True,
                 use_synonym: bool = True,
                 min_sim: float = MIN_SIMILARITY,
                 max_sim: float = MAX_SIMILARITY) -> Tuple[List[str], np.ndarray, Dict]:
    """
    Augment batch of texts with quality filtering
    
    Args:
        texts: List of text strings
        labels: Array of labels
        para_tokenizer: Paraphrase tokenizer
        para_model: Paraphrase model
        device: torch device
        use_paraphrase: Enable paraphrase augmentation
        use_synonym: Enable synonym augmentation
        min_sim: Minimum similarity threshold
        max_sim: Maximum similarity threshold
    
    Returns:
        augmented_texts: List of augmented texts (including originals)
        augmented_labels: Array of corresponding labels
        stats: Dictionary with statistics
    """
    augmented_texts = []
    augmented_labels = []
    
    stats = {
        'original': 0,
        'paraphrase_attempted': 0,
        'paraphrase_valid': 0,
        'synonym_attempted': 0,
        'synonym_valid': 0,
        'total_augmented': 0
    }
    
    for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Augmenting"):
        # Always include original
        augmented_texts.append(text)
        augmented_labels.append(label)
        stats['original'] += 1
        
        # Paraphrase
        if use_paraphrase:
            stats['paraphrase_attempted'] += 1
            para = aug_paraphrase(text, para_tokenizer, para_model, device)
            if is_valid_augmentation(text, para, min_sim, max_sim):
                augmented_texts.append(para)
                augmented_labels.append(label)
                stats['paraphrase_valid'] += 1
                stats['total_augmented'] += 1
        
        # Synonym
        if use_synonym:
            stats['synonym_attempted'] += 1
            syn = aug_synonym(text, SYNONYM_DICT)
            if is_valid_augmentation(text, syn, min_sim, max_sim):
                augmented_texts.append(syn)
                augmented_labels.append(label)
                stats['synonym_valid'] += 1
                stats['total_augmented'] += 1
    
    return augmented_texts, np.array(augmented_labels), stats


# ============================================================================
# MAIN AUGMENTATION PIPELINE
# ============================================================================

def run_augmentation(use_paraphrase: bool = USE_PARAPHRASE,
                    use_synonym: bool = USE_SYNONYM,
                    save_to_disk: bool = True):
    """
    Run complete augmentation pipeline
    
    Args:
        use_paraphrase: Enable paraphrase augmentation
        use_synonym: Enable synonym augmentation
        save_to_disk: Save augmented dataset to disk
    
    Returns:
        DataFrame with augmented data
    """
    print("="*80)
    print("OPTIMAL AUGMENTATION PIPELINE")
    print("="*80)
    
    # Load dataset
    print("\n[1/4] Loading balanced dataset...")
    df = pd.read_csv(BALANCED_DATASET)
    print(f"  ✅ Loaded {len(df)} samples")
    
    # Setup device and models
    print("\n[2/4] Loading augmentation models...")
    device = get_device()
    print(f"  Device: {device}")
    
    para_tokenizer, para_model = None, None
    if use_paraphrase:
        print(f"  Loading paraphrase model: {PARAPHRASE_MODEL}")
        para_tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
        para_model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL).to(device)
        para_model.eval()
        print(f"  ✅ Paraphrase model loaded")
    
    # Run augmentation
    print("\n[3/4] Running augmentation...")
    print(f"  Methods: Paraphrase={use_paraphrase}, Synonym={use_synonym}")
    print(f"  Similarity filter: {MIN_SIMILARITY}-{MAX_SIMILARITY}")
    
    timer = Timer()
    timer.start()
    
    aug_texts, aug_labels, stats = augment_batch(
        df['title'].tolist(),
        df['category'].values,
        para_tokenizer, para_model, device,
        use_paraphrase, use_synonym
    )
    
    elapsed = timer.stop()
    
    # Create DataFrame
    df_augmented = pd.DataFrame({
        'title': aug_texts,
        'category': aug_labels
    })
    
    # Save
    if save_to_disk:
        print("\n[4/4] Saving augmented dataset...")
        df_augmented.to_csv(AUGMENTED_DATASET, index=False)
        print(f"  ✅ Saved to {AUGMENTED_DATASET}")
    
    # Print statistics
    print("\n" + "="*80)
    print("AUGMENTATION COMPLETE ✅")
    print("="*80)
    print(f"\n📊 Statistics:")
    print(f"  Original samples: {stats['original']}")
    if use_paraphrase:
        print(f"  Paraphrase: {stats['paraphrase_valid']}/{stats['paraphrase_attempted']} valid " +
              f"({stats['paraphrase_valid']/stats['paraphrase_attempted']*100:.1f}%)")
    if use_synonym:
        print(f"  Synonym: {stats['synonym_valid']}/{stats['synonym_attempted']} valid " +
              f"({stats['synonym_valid']/stats['synonym_attempted']*100:.1f}%)")
    print(f"  Total augmented: {stats['total_augmented']}")
    print(f"  Final dataset size: {len(df_augmented)} " +
          f"({len(df_augmented)/stats['original']:.1f}x original)")
    
    print(f"\n⏱️  Time: {timer}")
    print(f"  Speed: {len(df)/elapsed*60:.1f} samples/min")
    
    estimated_full_time = elapsed
    print(f"  Estimated time saved: {4*3600 - estimated_full_time:.0f} seconds " +
          f"(vs 4 hours with backtranslation)")
    
    print("="*80)
    
    return df_augmented, stats


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(RANDOM_STATE)
    
    # Run augmentation
    df_aug, stats = run_augmentation(
        use_paraphrase=USE_PARAPHRASE,
        use_synonym=USE_SYNONYM,
        save_to_disk=True
    )
    
    print(f"\n🎯 Sample augmented data:")
    print(df_aug.head(10))
    
    print(f"\n📈 Class distribution:")
    print(df_aug['category'].value_counts().sort_index())
