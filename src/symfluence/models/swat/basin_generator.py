# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SWAT Basin File Generator

Generates the basin-level parameter file (.bsn), weather generator file (.wgn),
and auxiliary stub files (.pnd, .wus, .chm) required by SWAT.
"""
import logging
from typing import Dict

from .parameters import DEFAULT_PARAMS

logger = logging.getLogger(__name__)


class SWATBasinGenerator:
    """Generates basin-level SWAT input files.

    Args:
        preprocessor: Parent SWATPreProcessor instance providing access
            to config, logger, paths, and helper methods.
    """

    def __init__(self, preprocessor):
        self.pp = preprocessor

    def generate_basin_file(self) -> None:
        """Generate SWAT basin file (.bsn) matching readbsn.f read order.

        readbsn.f reads the .bsn file in a very specific order after 3
        title lines.  Every value must appear on its own line, in exactly
        the order that the Fortran reads consume them.  String values
        (petfile, wwqfile) use format 1000 = (a), reading the first
        len(variable) characters.  Integer values use format 1001 = (i4).
        Real values use free-format (read *).
        """
        bsn_path = self.pp.txtinout_dir / 'basins.bsn'

        # Default values for basin-level parameters
        sftmp = DEFAULT_PARAMS['SFTMP']
        smtmp = DEFAULT_PARAMS['SMTMP']
        smfmx = DEFAULT_PARAMS['SMFMX']
        smfmn = DEFAULT_PARAMS['SMFMN']
        timp = DEFAULT_PARAMS['TIMP']
        snocovmx = 1.0       # Minimum snow water content for 100% snow cover [mm]
        sno50cov = 0.5       # Snow water content for 50% snow cover [mm]
        ipet = 1             # PET method (0=P-M, 1=Hargreaves, 2=Priestley-Taylor)
        petfile = ""         # PET input file (blank = not used)
        esco_bsn = DEFAULT_PARAMS['ESCO']
        epco_bsn = 0.95      # Plant uptake compensation factor
        evlai = 3.0          # Leaf area index at which no evap from water surface
        ffcb = 0.0           # Initial soil water storage as fraction of FC

        ievent = 0           # Rainfall/runoff code (0=daily)
        icrk = 0             # Crack flow code
        surlag_bsn = DEFAULT_PARAMS['SURLAG']
        adj_pkr = 0.0        # Peak rate adjustment for sediment routing
        prf_bsn = 1.0        # Peak rate adjustment for channel sediment routing
        spcon_bsn = 0.0001   # Linear coeff for channel sed re-entrainment
        spexp_bsn = 1.0      # Exponent for channel sed re-entrainment

        rcn_sub_bsn = 1.0    # Concentration of N in rainfall [mg/l]
        cmn_bsn = 0.0003     # Rate coefficient for humus mineralization
        n_updis = 20.0       # N uptake distribution parameter
        p_updis = 20.0       # P uptake distribution parameter
        nperco_bsn = 0.20    # N percolation coefficient
        pperco_bsn = 10.0    # P percolation coefficient
        phoskd_bsn = 175.0   # P soil partitioning coefficient
        psp_bsn = 0.4        # P sorption coefficient
        rsdco = 0.05         # Residue decomposition coefficient

        percop = 0.5         # Pesticide percolation coefficient
        isubwq = 0           # Subbasin water quality code

        # 16 bacteria parameters (all 0.0)
        bact_params = [0.0] * 16
        ised_det = 0         # Sediment detachment method

        irte = 0             # Channel routing (0=variable-storage)
        msk_co1 = 0.0        # Muskingum calibration coeff 1
        msk_co2 = 3.5        # Muskingum calibration coeff 2
        msk_x = 0.2          # Muskingum weighting factor
        ideg = 0             # Channel degradation code
        iwq = 0              # In-stream water quality code
        wwqfile = "basins.wwq"  # Water quality file (char*13)
        trnsrch = 0.0        # Fraction of transmission losses to deep aquifer
        evrch = 1.0          # Reach evaporation coefficient
        irtpest = 0          # Pesticide routing flag
        icn = 0              # Daily CN calculation method (0=traditional)
        cncoef = 0.0         # CN coefficient (for icn=2)
        cdn_bsn = 1.4        # Denitrification exponential rate coefficient
        sdnco_bsn = 1.1      # Denitrification threshold water content
        bact_swf = 0.0       # Fraction of bacteria in solution

        # Optional params (read with iostat=eof)
        bactmx = 0.0
        bactminlp = 0.0
        bactminp = 0.0
        wdlprch = 0.0
        wdprch = 0.0
        wdlpres = 0.0
        wdpres = 0.0
        tb_adj = 0.0
        depimp_bsn = 6000.0
        ddrain_bsn = 0.0
        tdrain_bsn = 0.0
        gdrain_bsn = 0.0
        cn_froz = 0.000862

        lines = []

        # -- 3 title lines --
        lines.append(" Basin parameters -- SYMFLUENCE generated")
        lines.append(" SWAT basin (.bsn) file")
        lines.append(" ")

        # -- Block 1: snow, PET, evap params (read *, then title) --
        lines.append(f"  {sftmp:14.4f}    | SFTMP : Snowfall temperature [deg C]")
        lines.append(f"  {smtmp:14.4f}    | SMTMP : Snow melt base temperature [deg C]")
        lines.append(f"  {smfmx:14.4f}    | SMFMX : Max melt rate [mm/deg C/day]")
        lines.append(f"  {smfmn:14.4f}    | SMFMN : Min melt rate [mm/deg C/day]")
        lines.append(f"  {timp:14.4f}    | TIMP : Snow pack temperature lag factor")
        lines.append(f"  {snocovmx:14.4f}    | SNOCOVMX : Min snow for 100% cover [mm]")
        lines.append(f"  {sno50cov:14.4f}    | SNO50COV : Snow for 50% cover")
        # ipet: integer format (i4) -- first 4 chars
        lines.append(f"{ipet:4d}                | IPET : PET method code")
        # petfile: format (a) -- first len chars; blank = not used
        lines.append(f"{petfile:13s}     | PETFILE : PET input file")
        lines.append(f"  {esco_bsn:14.4f}    | ESCO : Soil evaporation compensation factor")
        lines.append(f"  {epco_bsn:14.4f}    | EPCO : Plant uptake compensation factor")
        lines.append(f"  {evlai:14.4f}    | EVLAI : LAI at which no evap from water surface")
        lines.append(f"  {ffcb:14.4f}    | FFCB : Initial soil water as fraction of FC")
        lines.append(" Runoff/Sediment:")

        # -- Block 2: runoff, sediment (read *, then title) --
        lines.append(f"{ievent:4d}                | IEVENT : Rainfall/runoff code")
        lines.append(f"{icrk:4d}                | ICRK : Crack flow code")
        lines.append(f"  {surlag_bsn:14.4f}    | SURLAG : Surface runoff lag coefficient")
        lines.append(f"  {adj_pkr:14.4f}    | ADJ_PKR : Peak rate adj for sediment routing")
        lines.append(f"  {prf_bsn:14.4f}    | PRF : Peak rate adj for channel sediment routing")
        lines.append(f"  {spcon_bsn:14.4f}    | SPCON : Linear coeff for channel sed re-entrainment")
        lines.append(f"  {spexp_bsn:14.4f}    | SPEXP : Exponent for channel sed re-entrainment")
        lines.append(" Nutrients:")

        # -- Block 3: nutrients (read *, then title) --
        lines.append(f"  {rcn_sub_bsn:14.4f}    | RCN : Concentration of N in rainfall [mg/l]")
        lines.append(f"  {cmn_bsn:14.4f}    | CMN : Rate coeff for humus mineralization")
        lines.append(f"  {n_updis:14.4f}    | N_UPDIS : N uptake distribution parameter")
        lines.append(f"  {p_updis:14.4f}    | P_UPDIS : P uptake distribution parameter")
        lines.append(f"  {nperco_bsn:14.4f}    | NPERCO : N percolation coefficient")
        lines.append(f"  {pperco_bsn:14.4f}    | PPERCO : P percolation coefficient")
        lines.append(f"  {phoskd_bsn:14.4f}    | PHOSKD : P soil partitioning coefficient")
        lines.append(f"  {psp_bsn:14.4f}    | PSP : P sorption coefficient")
        lines.append(f"  {rsdco:14.4f}    | RSDCO : Residue decomposition coefficient")
        lines.append(" Pesticides:")

        # -- Block 4: percop, then title, isubwq, then title --
        lines.append(f"  {percop:14.4f}    | PERCOP : Pesticide percolation coefficient")
        lines.append(" Algae/CBOD/DO:")
        lines.append(f"{isubwq:4d}                | ISUBWQ : Subbasin water quality code")
        lines.append(" Bacteria:")

        # -- 16 bacteria params (all 0), then ised_det, then title --
        bact_names = [
            "WDPQ", "WGPQ", "WDLPQ", "WGLPQ",
            "WDPS", "WGPS", "WDLPS", "WGLPS",
            "WDPF", "WGPF", "WDLPF", "WGLPF",
            "WDPSC", "WGPSC", "WDLPSC", "WGLPSC",
        ]
        for i, bname in enumerate(bact_names):
            lines.append(f"  {bact_params[i]:14.4f}    | {bname}")
        lines.append(f"{ised_det:4d}                | ISED_DET : Sediment detachment method")
        lines.append(" Channel Routing:")

        # -- Block 5: routing params (read *) --
        lines.append(f"{irte:4d}                | IRTE : Channel routing method")
        lines.append(f"  {msk_co1:14.4f}    | MSK_CO1 : Muskingum calib coeff 1")
        lines.append(f"  {msk_co2:14.4f}    | MSK_CO2 : Muskingum calib coeff 2")
        lines.append(f"  {msk_x:14.4f}    | MSK_X : Muskingum weighting factor")
        lines.append(f"{ideg:4d}                | IDEG : Channel degradation code")
        lines.append(f"{iwq:4d}                | IWQ : In-stream water quality code")
        # wwqfile: format (a), char*13
        lines.append(f"{wwqfile:13s}     | WWQFILE : Water quality input file")
        lines.append(f"  {trnsrch:14.4f}    | TRNSRCH : Fraction of trans losses to deep aq")
        lines.append(f"  {evrch:14.4f}    | EVRCH : Reach evaporation coefficient")
        lines.append(f"{irtpest:4d}                | IRTPEST : Pesticide routing flag")
        lines.append(f"{icn:4d}                | ICN : Daily CN calculation method")
        lines.append(f"  {cncoef:14.4f}    | CNCOEF : CN coefficient")
        lines.append(f"  {cdn_bsn:14.4f}    | CDN : Denitrification exponential rate coeff")
        lines.append(f"  {sdnco_bsn:14.4f}    | SDNCO : Denitrification threshold water content")
        lines.append(f"  {bact_swf:14.4f}    | BACT_SWF : Fraction of bacteria in solution")

        # -- Optional params (read with iostat=eof) --
        lines.append(f"  {bactmx:14.4f}    | BACTMX")
        lines.append(f"  {bactminlp:14.4f}    | BACTMINLP")
        lines.append(f"  {bactminp:14.4f}    | BACTMINP")
        lines.append(f"  {wdlprch:14.4f}    | WDLPRCH")
        lines.append(f"  {wdprch:14.4f}    | WDPRCH")
        lines.append(f"  {wdlpres:14.4f}    | WDLPRES")
        lines.append(f"  {wdpres:14.4f}    | WDPRES")
        lines.append(f"  {tb_adj:14.4f}    | TB_ADJ")
        lines.append(f"  {depimp_bsn:14.4f}    | DEPIMP_BSN")
        lines.append(f"  {ddrain_bsn:14.4f}    | DDRAIN_BSN")
        lines.append(f"  {tdrain_bsn:14.4f}    | TDRAIN_BSN")
        lines.append(f"  {gdrain_bsn:14.4f}    | GDRAIN_BSN")
        lines.append(f"  {cn_froz:14.6f}    | CN_FROZ")

        bsn_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Basin file written: {bsn_path}")

        # Write dummy basins.wwq file (required by readbsn.f when iwq=0)
        wwq_path = self.pp.txtinout_dir / 'basins.wwq'
        wwq_path.write_text(
            " Water quality file -- SYMFLUENCE\n", encoding='utf-8')
        logger.info(f"Water quality file written: {wwq_path}")

    # ------------------------------------------------------------------
    # Auxiliary file generators referenced by the .sub file
    # ------------------------------------------------------------------

    def generate_wgn_file(self, props: Dict) -> None:
        """Generate a weather generator file (.wgn) for the sub-basin.

        ``readwgn.f`` reads the file on unit 114 with these formats:

        - Line 1: title (format a)
        - Line 2: station latitude  (format 12x,f7.2)
        - Line 3: station elevation (format 12x,f7.2)
        - Line 4: rain_yrs          (format 12x,f7.2)
        - Lines 5-18: 14 rows of 12 monthly values (format 12f6.2,
          except line 9 which uses 12f6.1 for pcpmm)

        The 14 monthly-data rows are (in order):
          TMPMX, TMPMN, TMPSTDMX, TMPSTDMN, PCPMM, PCPSTD, PCPSKW,
          PR_W1, PR_W2, PCPD, RAINHHMX, SOLARAV, DEWPT, WNDAV
        """
        wgn_path = self.pp.txtinout_dir / '000010001.wgn'

        lat = props.get('lat', 51.0)
        elev = props.get('elev', 1000.0)

        # Reasonable monthly climate normals for a mid-latitude mountain
        # catchment (e.g. Canadian Rockies ~51N).  These are placeholder
        # values that keep the weather generator stable.
        # fmt: off
        tmpmx  = [ -5.0, -2.0,  3.0, 10.0, 16.0, 20.0, 23.0, 22.0, 17.0, 10.0,  2.0, -4.0]
        tmpmn  = [-15.0,-13.0, -8.0, -2.0,  3.0,  7.0,  9.0,  8.0,  4.0, -1.0, -8.0,-14.0]
        tmpstdmx = [6.0,  6.0,  5.5,  5.0,  4.5,  4.0,  3.5,  3.5,  4.5,  5.0,  5.5,  6.0]
        tmpstdmn = [6.5,  6.5,  5.5,  4.5,  3.5,  3.0,  2.5,  2.5,  3.5,  4.5,  5.5,  6.5]
        pcpmm  = [30.0, 25.0, 30.0, 40.0, 55.0, 70.0, 65.0, 55.0, 45.0, 35.0, 30.0, 30.0]
        pcpstd = [ 5.0,  4.5,  5.0,  6.0,  7.0,  8.0,  8.5,  7.5,  6.5,  5.5,  5.0,  5.0]
        pcpskw = [ 1.5,  1.5,  1.5,  1.2,  1.0,  0.8,  0.8,  1.0,  1.2,  1.5,  1.5,  1.5]
        pr_w1  = [ 0.15, 0.15, 0.18, 0.22, 0.25, 0.28, 0.25, 0.22, 0.20, 0.18, 0.15, 0.15]
        pr_w2  = [ 0.55, 0.55, 0.58, 0.60, 0.62, 0.65, 0.62, 0.60, 0.58, 0.55, 0.55, 0.55]
        pcpd   = [ 8.0,  7.0,  9.0, 10.0, 12.0, 13.0, 11.0, 10.0,  9.0,  8.0,  8.0,  8.0]
        rainhhmx = [10.0, 10.0, 12.0, 15.0, 20.0, 25.0, 28.0, 25.0, 20.0, 15.0, 12.0, 10.0]
        solarav = [ 6.0,  9.0, 13.0, 17.0, 20.0, 22.0, 22.0, 19.0, 15.0, 10.0,  7.0,  5.0]
        dewpt  = [-18.0,-16.0,-11.0, -4.0,  1.0,  5.0,  7.0,  6.0,  2.0, -3.0,-10.0,-17.0]
        wndav  = [ 3.5,  3.5,  3.8,  4.0,  3.5,  3.2,  2.8,  2.8,  3.0,  3.5,  3.5,  3.5]
        # fmt: on

        def fmt_12f62(vals):
            """Format 12 monthly values in Fortran 12f6.2."""
            return ''.join(f"{v:6.2f}" for v in vals)

        def fmt_12f61(vals):
            """Format 12 monthly values in Fortran 12f6.1."""
            return ''.join(f"{v:6.1f}" for v in vals)

        lines = [
            " Weather generator data -- SYMFLUENCE generated",
            f"  LATI (deg){lat:7.2f}",
            f"  ELEV (m)  {elev:7.2f}",
            f"  RAIN_YRS  {10.00:7.2f}",
            fmt_12f62(tmpmx),       # TMPMX
            fmt_12f62(tmpmn),       # TMPMN
            fmt_12f62(tmpstdmx),    # TMPSTDMX
            fmt_12f62(tmpstdmn),    # TMPSTDMN
            fmt_12f61(pcpmm),       # PCPMM (format 12f6.1)
            fmt_12f62(pcpstd),      # PCPSTD
            fmt_12f62(pcpskw),      # PCPSKW
            fmt_12f62(pr_w1),       # PR_W1
            fmt_12f62(pr_w2),       # PR_W2
            fmt_12f62(pcpd),        # PCPD
            fmt_12f62(rainhhmx),    # RAINHHMX
            fmt_12f62(solarav),     # SOLARAV
            fmt_12f62(dewpt),       # DEWPT
            fmt_12f62(wndav),       # WNDAV
        ]
        wgn_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Weather generator file written: {wgn_path}")

    def generate_pnd_stub(self) -> None:
        """Generate a minimal pond/wetland stub file (.pnd).

        ``readpnd.f`` reads every value with ``iostat=eof``, so an
        empty or minimal file is safe -- EOF stops the reads
        gracefully and all pond variables retain their default (zero)
        values.
        """
        pnd_path = self.pp.txtinout_dir / '000010001.pnd'
        lines = [
            " Pond/Wetland data -- SYMFLUENCE stub (no ponds/wetlands)",
            " Pond data:",
        ]
        pnd_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Pond stub file written: {pnd_path}")

    def generate_wus_stub(self) -> None:
        """Generate a minimal water-use stub file (.wus).

        ``readwus.f`` reads with ``iostat=eof``; an empty/minimal
        file is safe.
        """
        wus_path = self.pp.txtinout_dir / '000010001.wus'
        lines = [
            " Water Use data -- SYMFLUENCE stub (no external water use)",
        ]
        wus_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Water use stub file written: {wus_path}")

    def generate_chm_stub(self) -> None:
        """Generate a minimal soil-chemistry stub file (.chm).

        ``hruallo.f`` opens the .chm file and reads groups of 11
        header lines then pesticide IDs until EOF.  A minimal file
        with just a title line causes the ``iostat=eof`` to fire
        immediately, which is the correct behaviour for a model with
        no pesticides.
        """
        chm_path = self.pp.txtinout_dir / '000010001.chm'
        lines = [
            " Soil chemistry data -- SYMFLUENCE stub (no pesticides)",
        ]
        chm_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.info(f"Soil chemistry stub file written: {chm_path}")
