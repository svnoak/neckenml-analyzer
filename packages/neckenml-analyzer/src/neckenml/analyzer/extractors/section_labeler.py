class ABSectionLabeler:
    def __init__(self, sr=16000, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def _extract_fingerprint(self, audio):
        import numpy as np
        import librosa
        # Handle empty or too short audio chunk
        if len(audio) < 512: return np.zeros(26)
        
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        if mfcc.shape[1] < 2: return np.zeros(26) # Safety for tiny segments
            
        delta = librosa.feature.delta(mfcc)
        fp = np.concatenate([mfcc.mean(axis=1), delta.mean(axis=1)])
        return fp

    def label_sections(self, audio, sections):
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.cluster import AgglomerativeClustering

        # --- SAFETY CHECK 1: Not enough sections to cluster ---
        if not sections or len(sections) < 2:
            return ["A"] * len(sections) if sections else []

        fingerprints = []

        for i in range(len(sections)):
            start = int(sections[i] * self.sr)
            end = int((sections[i+1] if i+1 < len(sections) else len(audio)/self.sr) * self.sr)
            
            # Avoid crash on empty segments
            if end <= start: 
                fingerprints.append(np.zeros(26))
                continue

            chunk = audio[start:end]
            fingerprints.append(self._extract_fingerprint(chunk))

        fingerprints = np.array(fingerprints)

        # --- SAFETY CHECK 2: Ensure fingerprints valid ---
        if len(fingerprints) < 2:
             return ["A"] * len(sections)

        # Hierarchical clustering
        n_clusters = min(4, len(fingerprints))
        
        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='euclidean',
                linkage='ward'
            ).fit(fingerprints)

            cluster_ids = clustering.labels_
        except Exception as e:
            print(f"[ERROR] Clustering failed: {e}")
            # Fallback: Just label them sequentially A, B, C...
            return [chr(65 + i) for i in range(len(sections))]

        # We'll just assign letters since that's common in folkmusic
        labels = []
        base_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        cluster_to_letter = {}
        next_letter_idx = 0
        cluster_examples = {}

        for idx, cid in enumerate(cluster_ids):
            if cid not in cluster_to_letter:
                cluster_to_letter[cid] = base_letters[next_letter_idx]
                next_letter_idx += 1

            letter = cluster_to_letter[cid]

            if cid not in cluster_examples:
                cluster_examples[cid] = fingerprints[idx]
                labels.append(letter)
            else:
                # Check for variants (A')
                rep_fp = cluster_examples[cid]
                try:
                    similarity = cosine_similarity([rep_fp], [fingerprints[idx]])[0][0]
                    if similarity > 0.90:
                        labels.append(letter)
                    else:
                        labels.append(letter + "'")
                except:
                    labels.append(letter)

        return labels