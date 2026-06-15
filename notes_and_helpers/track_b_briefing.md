# Track B — Known-Geometry NNLS für Stage 2

Selbsttragendes Briefing für eine separate Claude-Code-Instanz. Track A
(DOA-Grid + CLEAN-SC) wird parallel in einer anderen Instanz bearbeitet —
deine Aufgabe ist Track B, vollständig isoliert davon.

## Kontext

**Repo:** `/home/steffen/Code/marty-mic-fly` (sollte bei dir der gleiche Pfad
sein). Branch `master`, Basis-Commit `b60ffa8`. Vor Beginn deiner Arbeit:
`git log --oneline -3` ausführen, oberster Commit muss
`refactor(stage2): factory dispatch hook for Track A/B subclasses` sein.

**Projekt:** MartyMicFly. AP2-A Stage-2 entfaltet das Mic-Array via CLEAN-SC,
maskiert die Drohnen-Anteile im rekonstruierten CSM und gibt einen residual
CSM zurück. Aktuelle Schwäche: bei tiefen Frequenzen (λ ≳ Apertur) lokalisiert
CLEAN-SC die Quelle nicht zuverlässig im 3D-Grid → die Subtraktion zieht im
ext-only-Test bis zu 246 dB CSM-Trace ab, weil sich die Energie über das
ganze Grid schmiert.

**Track B Ziel:** alternativen Stage-2-Algorithmus implementieren, der **nicht
nach Quellpositionen sucht**, sondern auf einem bekannten Atom-Set (Rotoren +
Zielposition) per NNLS pro Frequenzbin die Quellpowers fittet. Eingebaut
neben CLEAN-SC; switchbar via YAML.

## Was bereits steht (nicht ändern)

### Vier-Phasen-Architektur in `src/martymicfly/processing/array_filter.py`

`ArrayFilterStage.process()` ist in vier überschreibbare Phasen geteilt:

```
csm, freqs   = self._build_csm(ctx)                    # Phase 1
fit_input    = self._build_fit_input(ctx)              # Phase 2 — Suchraum
source_map   = self._fit_powers(csm, freqs, fit_input, ctx)  # Phase 3 — Algo
masks        = self._build_masks(fit_input, ctx)       # Phase 4 — Maske
# danach: reconstruct_csm + residual_csm + beam_maps + steer_to_psd + metadata
```

`FitInput`-Dataclass im selben File: `positions: (G,3)`, `reshape_hint: tuple`,
`aux: dict`. `MaskBundle`: `active: (G,) bool`, `named: dict[str, (G,) bool]`.

### Factory-Dispatch in `array_filter.py` am Dateiende

```python
def _array_filter_factory(cfg, **kwargs):
    if getattr(cfg, "doa_grid", None) is not None:
        from martymicfly.processing.array_filter_doa import DoaArrayFilterStage
        return DoaArrayFilterStage(cfg)
    if getattr(cfg, "algorithm", None) == "known_geometry_lsq":
        from martymicfly.processing.array_filter_atoms import KnownAtomsArrayFilterStage
        return KnownAtomsArrayFilterStage(cfg)
    return ArrayFilterStage(cfg)
register_stage_builder("array_filter", _array_filter_factory)
```

Der Dispatch ruft **deine** neue Klasse `KnownAtomsArrayFilterStage` auf,
sobald `cfg.algorithm == "known_geometry_lsq"`. Du musst nichts an dieser
Funktion ändern.

### Algorithmus-Registry: `src/martymicfly/processing/algorithms/`

`__init__.py` exponiert `ALGORITHM_REGISTRY: dict[str, type[Algorithm]]`,
`register_algorithm(cls)` als Decorator. CLEAN-SC ist dort bereits registriert.

### SourceMap-Kontrakt in `src/martymicfly/processing/algorithms/base.py`

```python
@dataclass(frozen=True)
class SourceMap:
    positions: np.ndarray       # (G, 3)
    powers: np.ndarray          # (F, G), reell ≥ 0
    frequencies: np.ndarray     # (F,)
    grid_shape: Optional[tuple[int, int]]  # bei dir None
    metadata: dict
    def subset(self, mask): ...
```

`reconstruct_csm(source_map, mic_positions)` baut den CSM zurück per
1/(4π·r) · exp(−j·2π·f·r/c) — *das* ist das Vorwärtsmodell, das du in
deinem Algorithmus konsistent nutzen musst.

### Vergleichstool und Referenz-Daten

- `python -m martymicfly.cli.compare_modes --config <yaml>` lädt einen
  Pipeline-YAML, fittet einmal, evaluiert alle drei Maskenmodi, druckt eine
  kombinierte Tabelle inkl. CLEAN-SC-Lokalisierungsstats.
- `python -m martymicfly.cli.probe_localization --config <yaml>` druckt nur
  die Lokalisierungstabelle.
- Ext-only Synth-File:
  `/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_external_only_gaptip.h5`
  enthält ausschließlich die externe Quelle bei z=−1.5 m, keine Drohne. Damit
  ist jede Subtraktion ein false-positive — perfekter Methodik-Floor-Test.
- Mixed Synth-File:
  `/media/steffen/Data/Arbeit/MartyMicFly/Messdaten/synth_data/ap2a_synth_mixed_gaptip.h5`
  Drohne + externe Quelle.
- Pipeline-Configs: `configs/pipeline_external_only_{target_box,drone_box,rotor_disc}.yaml`
  und `configs/example_pipeline.yaml` (mixed).

## Deine Deliverables

### 1. Algorithmus-Modul `src/martymicfly/processing/algorithms/known_geometry_lsq.py`

```python
"""NNLS auf bekanntem Atom-Set: CSM-Fitting ohne Grid-Suche.

Vorwärtsmodell pro Frequenz f:
    C_obs[f] = sum_g P[f, g] * a_g(f) * a_g(f)^H

mit:
    a_g(f)[m] = (1 / (4π·r_{m,g})) * exp(-j·2πf/c · r_{m,g})

Vektorisiert: D[f] · p[f] ≈ vec(C_obs[f]), gelöst als NNLS pro f-Bin.
"""
from martymicfly.processing.algorithms import register_algorithm
from martymicfly.processing.algorithms.base import Algorithm, SourceMap
from martymicfly.constants import SPEED_OF_SOUND
from scipy.optimize import nnls
import numpy as np

@register_algorithm
class KnownGeometryLsqAlgorithm:
    name = "known_geometry_lsq"
    consumes = "csm"

    def fit(self, *, csm, frequencies, time_data, sample_rate,
            mic_positions, grid_positions, params):
        # grid_positions hier == Atom-Positionen, (G, 3) klein (5..17)
        c = float(params.get("speed_of_sound", SPEED_OF_SOUND))
        ridge = float(params.get("ridge", 0.0))   # optionale Tikhonov
        ...
        # Steeringmatrix H[f] = (1 / (4π·r_{m,g})) * exp(-j·2πf·r/c), (M, G)
        # Wörterbuch D[f] = stack(real, imag) von vec(h_g h_g^H), (2M², G)
        # y[f] = stack(real, imag) von vec(csm[f]), (2M²,)
        # Optional: Identity-Atom für diffuses Rauschen (params['include_diffuse'])
        # Pro Frequenzbin: nnls(D[f], y[f]) → p[f] (G,)
        # SourceMap zurückgeben
        return SourceMap(positions=grid_positions, powers=powers,
                          frequencies=frequencies, grid_shape=None,
                          metadata={"algorithm": "known_geometry_lsq"})
```

**Wichtig:** Das Vorwärtsmodell `1/(4π·r) · exp(−j·2πf·r/c)` muss mit
`reconstruct_csm` in `base.py` exakt übereinstimmen — sonst stimmt die
nachgelagerte CSM-Subtraktion nicht.

**Vektorisierung-Hinweis:** Komplexe NNLS gibt es nicht, Real- und
Imaginärteil von vec(C_obs) und vec(h h^H) müssen gestapelt werden:

```python
H = (1.0 / (4.0 * np.pi * r)) * np.exp(-2j * np.pi * f * r / c)   # (M, G)
hh = np.einsum('mg,ng->mng', H, H.conj())                          # (M, M, G)
D_complex = hh.reshape(-1, G)                                       # (M*M, G)
D_real = np.concatenate([D_complex.real, D_complex.imag], axis=0)   # (2M*M, G)
y_complex = csm[fi].reshape(-1)                                     # (M*M,)
y_real = np.concatenate([y_complex.real, y_complex.imag])           # (2M*M,)
p_f, _ = nnls(D_real, y_real)
```

Bei niedrigen Frequenzen kann D[f] schlecht konditioniert sein. Pragmatischer
Schutz: wenn `cond(D) > 1e10` → `p[f] = 0`. Optional Tikhonov:
`D' = stack(D, sqrt(α)·I_G)`, `y' = stack(y, 0_G)`.

### 2. Config-Erweiterung in `src/martymicfly/config.py`

`ArrayFilterStageConfig.algorithm` erweitern auf
`Literal["clean_sc", "known_geometry_lsq"]`. Außerdem optionalen
Atom-Set-Block hinzufügen:

```python
class AtomSetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drone_atoms: Literal["rotor_positions", "subsource_positions"] = "rotor_positions"
    target_atom_position_m: tuple[float, float, float] | None = None  # default: target_point_m
    include_diffuse: bool = False
    ridge: float = 0.0
    cond_threshold: float = 1e10

class ArrayFilterStageConfig(BaseModel):
    # ... existing fields ...
    atoms: AtomSetConfig | None = None  # nur relevant für algorithm=known_geometry_lsq
```

**Konflikt-Hinweis:** Track A fügt parallel `doa_grid: DoaGridConfig | None`
zu derselben Klasse hinzu. Wenn beim Merge ein Konflikt entsteht, ist er
trivial: zwei zusätzliche Felder, beide additiv.

### 3. Stage-Subclass `src/martymicfly/processing/array_filter_atoms.py`

```python
from dataclasses import field
import numpy as np
from martymicfly.processing.array_filter import (
    ArrayFilterStage, FitInput, MaskBundle,
)

class KnownAtomsArrayFilterStage(ArrayFilterStage):
    """Stage-2 Variante: NNLS auf bekanntem Atom-Set statt Grid-Suche."""

    def _build_fit_input(self, ctx) -> FitInput:
        plat = ctx.metadata["platform"]
        rotor_positions = np.asarray(plat["rotor_positions"]).T  # (R, 3)
        # Atome zusammenstellen: alle Rotoren + Zielposition
        atoms = [(*rotor_positions[i], "drone") for i in range(rotor_positions.shape[0])]
        target_pos = (self.cfg.atoms.target_atom_position_m if self.cfg.atoms and self.cfg.atoms.target_atom_position_m
                      else self.cfg.target_point_m)
        atoms.append((*target_pos, "target"))
        # Optional: identity-Atom (handled in algorithm via params)
        positions = np.array([a[:3] for a in atoms], dtype=np.float64)
        kinds = np.array([a[3] for a in atoms])
        return FitInput(positions=positions, reshape_hint=(positions.shape[0],),
                        aux={"atom_kinds": kinds})

    def _fit_powers(self, csm, freqs, fit_input, ctx):
        # Algorithmus erhält Atompositionen als grid_positions
        sm = self.algo.fit(
            csm=csm, frequencies=freqs, time_data=None,
            sample_rate=ctx.sample_rate, mic_positions=ctx.mic_positions,
            grid_positions=fit_input.positions,
            params={
                "include_diffuse": (self.cfg.atoms.include_diffuse if self.cfg.atoms else False),
                "ridge": (self.cfg.atoms.ridge if self.cfg.atoms else 0.0),
                "cond_threshold": (self.cfg.atoms.cond_threshold if self.cfg.atoms else 1e10),
            },
        )
        # Trace-rescale ist hier konzeptionell nicht nötig (NNLS fittet direkt
        # in Pa², kein Acoular-Skalierungsfehler), aber zur Sicherheit dasselbe
        # Verhalten wie CLEAN-SC: identischer Output bei korrektem Modell.
        return sm

    def _build_masks(self, fit_input, ctx) -> MaskBundle:
        kinds = fit_input.aux["atom_kinds"]
        drone_atom_mask = (kinds == "drone")
        target_atom_mask = (kinds == "target")
        return MaskBundle(
            active=drone_atom_mask,
            named={
                "drone_atom_mask": drone_atom_mask,
                "target_atom_mask": target_atom_mask,
            },
        )
```

`integrate_band_maps` in array_filter.py erwartet aktuell `(nx, ny, nz)`-shape
und ruft `nx, ny, nz = grid_shape`. Für deinen Atom-Fall scheitert das
unpacking. Lösungswege (du wählst den eleganteren):
- (a) `integrate_band_maps` flexibler machen: wenn `grid_shape` 1-Tupel ist,
  nicht reshapen.
- (b) `reshape_hint=(G, 1, 1)` zurückgeben — das integriert technisch sauber.

### 4. Pipeline-Configs

Drei neue YAMLs nach dem Muster `configs/pipeline_external_only_*.yaml`,
diesmal mit `algorithm: known_geometry_lsq` und `atoms:`-Block:

- `configs/pipeline_external_only_nnls.yaml` (gegen das ext-only Synth-File)
- `configs/pipeline_mixed_nnls.yaml` (gegen das mixed Synth-File)

Stage-1 (notch) im ext-only-Fall weglassen wie in den existierenden
ext-only-Configs.

### 5. Tests `tests/test_known_geometry_lsq.py`

Mindestens:
- **Unit:** Single-Atom-Setup mit synthetischem CSM `c = a·aᴴ * P`. NNLS muss
  P am richtigen Atom finden, alle anderen ≈ 0.
- **Unit:** Multi-Atom (Rotor + Target), incoherent, NNLS muss beide trennen.
- **End-to-End** auf `tests/fixtures/tiny_synth_mixed.h5` + 4-Mic-Geom: Stage
  läuft durch, residual_csm ist hermitesch, target_psd_post finit, ≥ 0
  band-integriert.
- Vergleich shape `KnownAtomsArrayFilterStage` vs. `ArrayFilterStage`:
  identisches metadata-Schema (csm_pre, residual_csm, frequencies, source_map,
  drone_mask, target_psd_pre/post, beam_maps, diagnostic_grid).

Existierende Tests dürfen **nicht** brechen. `uv run pytest` muss am Ende
deiner Arbeit grün sein.

## Akzeptanzkriterien

1. `uv run pytest` zeigt **alle** Tests grün (90 vorher + deine neuen).
2. `python -m martymicfly.cli.compare_modes --config configs/pipeline_external_only_nnls.yaml`
   läuft fehlerfrei durch (auch wenn die Tabelle für NNLS noch dürftig
   bestückt aussieht — nur drone_atom-Subtraktion existiert, keine
   target_box-Variante).
3. Smoke gegen ext-only:
   ```
   python -m martymicfly.cli.run_pipeline --config configs/pipeline_external_only_nnls.yaml
   ```
   muss `metrics.json` erzeugen mit endlichen Werten.
4. Auf dem ext-only File **erwartet das Modell**, dass:
   - dem Target-Atom bei (0, 0, −1.5) der Großteil der Power zugewiesen wird
   - den Rotor-Atomen (z ≈ 0) Power ≈ 0 zugewiesen wird
   - daraus: `csm_red` im ext-only nahe 0 dB in **allen** Bändern (vs. 246 dB
     bei target_box-CLEAN-SC im low-band — *das* ist der Methodik-Floor-Win)
5. Auf dem mixed File: target_psd_post sollte näher an der GT-PSD liegen als
   die CLEAN-SC-Variante, vor allem im low-band.

## Was du NICHT anrühren sollst

- `src/martymicfly/processing/algorithms/base.py` (SourceMap, reconstruct_csm)
- `src/martymicfly/processing/array_filter.py` (außer dem Factory-Dispatch
  bereits da, der dich automatisch aufruft)
- `src/martymicfly/eval/array_metrics.py`
- `src/martymicfly/eval/array_plots.py`
- `src/martymicfly/cli/run_pipeline.py`
- `src/martymicfly/cli/compare_modes.py`
- Existierende Tests
- Track A's geplante Files (`array_filter_doa.py`, `DoaGridConfig`)

## Commit-Strategie

Auf einem Feature-Branch `track-b-nnls`:
```
git checkout -b track-b-nnls
# arbeiten, mehrere kleine Commits OK
git push origin track-b-nnls   # nur wenn Steffen explizit darum bittet
```

Steffen merged zurück auf master. Du musst nicht selbst mergen.

## Empfohlener Arbeitsablauf

1. Lies `src/martymicfly/processing/array_filter.py` (Phase-Architektur) und
   `src/martymicfly/processing/algorithms/base.py` (SourceMap, reconstruct_csm).
2. Lies einen existierenden Algorithmus als Vorbild:
   `src/martymicfly/processing/algorithms/clean_sc.py`.
3. Schreibe `known_geometry_lsq.py` mit minimalem Atom-Setup (4 Rotoren + 1
   Target) und teste an einem Single-Frequenz-Single-Atom-Synthetic-CSM.
4. Sobald das passt: Stage-Subclass + Config-Felder + ein einziger
   Pipeline-Run gegen das ext-only-Fixture.
5. Erst dann gegen die echten Synth-Files.
6. Tests schreiben, alle grün ziehen.
7. Final-Commit-Reihe auf `track-b-nnls`, Steffen Bescheid geben.

Bei Fragen die hier nicht beantwortet sind: lies den entsprechenden Code,
**rate nicht**.
