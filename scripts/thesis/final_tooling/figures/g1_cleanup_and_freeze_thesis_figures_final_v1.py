#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path(r"C:\humanoid_rl_lab\logs\thesis\figures_final")
PASS_MARKER = "G1_THESIS_FIGURES_FINAL_PACKAGE_FROZEN_PASS"


# fig_id, folder, stem, data_rel, png_sha, pdf_sha, data_sha
CANONICAL = [
    (
        "F01a", "01_v2_v3_evolution", "F01a_v2v3_straight_tracking_error_v4",
        "01_v2_v3_evolution/F01a_v2v3_straight_tracking_error_data_v4.csv",
        "D840DE0375022C9C39DF8149FF21B116B43500C982A32C5A0E6A572C9612F1E0",
        "D4E841927423387A3A8B254E77FF71783567F16538B1AEE68D0599CF894C10B9",
        "5F8D83A41E28E54BD97EA82E6AE0B4ABFA0174AE4355EB8DB3FB87B8ACABFDCF",
    ),
    (
        "F01b", "01_v2_v3_evolution", "F01b_v2v3_straight_mean_tilt_v2",
        "01_v2_v3_evolution/F01b_v2v3_straight_mean_tilt_data_v2.csv",
        "037556901C0DEFE34CB5BD62B742FA08A162D9BD28A6CA7340D247B8EF869D6A",
        "D5E864B86AAC6BA565BD43ECA61AE4AB1E6FF7911FDB077420F7B93DB098E6A6",
        "43ABB668D9F1DC04F093695EF330BC9B8566546606E3678E0BA8DC744764D2D4",
    ),
    (
        "F01c", "01_v2_v3_evolution", "F01c_v2a_vs_v2f_yaw_tracking_v2",
        "01_v2_v3_evolution/F01c_v2a_vs_v2f_yaw_tracking_data_v2.csv",
        "652452236AFFF4112FEC1372B73C92C5C3E02992262F05760F8623DAFCF7341B",
        "1368F2C6BF93F229D493F1F5BCA55A886469E1F1562B3BFF72B4C1F63B6F158F",
        "4BCDDBDCC25CCAB9A93C23F3FDCC17EC16ED071690CA5FFFEBAAE533E65AFACB",
    ),
    (
        "F02a", "02_v4_campaign", "F02a_v4_training_episode_length_v3",
        "02_v4_campaign/data/F02a_v4_training_episode_length_data_v3.csv",
        "31B46E1287CE1D7C0322C173E40FD72C288D012127D04C36949004991407D9C7",
        "344C8966760D66731BBE86A110B219DFE4D0F61732F74CF0DCD5D4C4B86ECEB2",
        "499E341FD52B3F5D3E4A8E772E07B4950F360DDC862A12A8A7075025C12F184E",
    ),
    (
        "F02a-Z", "02_v4_campaign", "F02a_v4_training_episode_length_zoom_v2",
        "02_v4_campaign/data/F02a_v4_training_episode_length_zoom_data_v2.csv",
        "0A3D030488F8DAB990B922ED3D1DB72F6EA2F9727A970D4A5A14D75F3DED5467",
        "BB9429D3BB746970F306B95DA86AB26F2CFE1215A01A4E7D64F96A5A1F109B2C",
        "499E341FD52B3F5D3E4A8E772E07B4950F360DDC862A12A8A7075025C12F184E",
    ),
    (
        "F02b", "02_v4_campaign", "F02b_v4b_v4n_cross_track_drift_v3",
        "02_v4_campaign/data/F02b_v4b_v4n_cross_track_drift_data_v3.csv",
        "8E26361D231BF475A9BB415A577C2EC21E7667A3756C74C485B19019D6765931",
        "CD5824FA010B269340A8AD95FAA4BE15621D45BDE6C7A55B81599D8BE7E6033F",
        "812A243F737B9E701AA2993EA24F649E84CA2351196309F5EACA4E85F95F04BF",
    ),
    (
        "F02c", "02_v4_campaign", "F02c_v4b_v4n_knee_kinematic_asymmetry_v3",
        "02_v4_campaign/data/F02c_v4b_v4n_knee_kinematic_asymmetry_data_v3.csv",
        "38A1D7A0DB7AB47B395786917C200A3CF9090B3A22E2354FE7B9BD05488F8E2B",
        "C3140BAE433C2074C29CD19803D490129C82223E306E7DE3772EC08C8CB56669",
        "FD770EC9FFAE93E06E109095A591033DE0DBAA81FE86DEAEC7E7027C7A6BA1CE",
    ),
    (
        "F03a", "03_v4n_final_gait", "F03a_v4n_gait_cycle_foot_clearance_contact_v3",
        "03_v4n_final_gait/data/F03a_v4n_gait_cycle_foot_clearance_contact_data_v3.csv",
        "60AF0599BFE49D403FFA06171644F60B2CCBA65BE4C383021CBB3B7DAAC5046D",
        "EB91D25AA8006613D420BC64CC30BD5F648BF6AE534620935BACB41D3AA0755C",
        "1F3C1600BF7A673EA72E72412A6FAADD57FCF66CAE5730FD84AD03F7FE171818",
    ),
    (
        "F03b", "03_v4n_final_gait", "F03b_v4n_forward_velocity_tracking_v2",
        "03_v4n_final_gait/data/F03b_v4n_forward_velocity_tracking_data_v2.csv",
        "75CFB0A4D418ECFB9BC276779AEB4E3C983F6AC79590E76F8A8B123FE18EFFBD",
        "99735D7C7A75C97D136EB7A7044B2253BDC5531D46F2FBC091539A385C860562",
        "EFFCE9B04EC5FE9417DBE8F7B33B5CE4CDA3384537732BBD1742CE5053E3D49B",
    ),
    (
        "F03c", "03_v4n_final_gait", "F03c_v4n_body_tilt_stability_v1",
        "03_v4n_final_gait/data/F03c_v4n_body_tilt_stability_data_v1.csv",
        "282B72441F973F20625DD9161E187B191FC7E6498A6BDBB5FB1BE898B77D3DC4",
        "6F8DF9C135D65F9F2FC723523CA5DDCA8582A6CE19291218939C8D4A04B797DE",
        "DE7CA636CB161E61D478D0E7449D87DFD0522308052916E50486A971510B8475",
    ),
    (
        "F04a", "04_v2g_final_stand", "F04a_v2g_standing_planar_trajectory_v3",
        "04_v2g_final_stand/data/F04a_v2g_standing_planar_trajectory_data_v3.csv",
        "1EB4A423246C184310A11110EDE8ABE4492D9FFF3782C163BB3DDD931A54C525",
        "1F6A5A7FDE75748192CA2590E36601821AE1AAE4B9D96913E9E1E2E018AC2D4E",
        "B9AEF6F7EA854F6F5B846D32EC2C91E318101A4BAE1D65A9021094ABD708586F",
    ),
    (
        "F04b", "04_v2g_final_stand", "F04b_v2g_standing_body_tilt_v2",
        "04_v2g_final_stand/data/F04b_v2g_standing_body_tilt_data_v2.csv",
        "C805E36C658F8A51E10B0D5400969BF1E03C31E57C0BFE876FECA01AE644534C",
        "D4A3A8734F6F8659AC647F62425910089B3042880B535CED8B99ADEAF31939B0",
        "3C12E2F209FF0F3A97EC24EF97DE51C2A34C6290156A56C725623C880611E550",
    ),
    (
        "F05a", "05_low_speed_tradeoff", "F05a_v2g_v4n_low_speed_tracking_tradeoff_v1",
        "05_low_speed_tradeoff/data/F05a_v2g_v4n_low_speed_tracking_tradeoff_data_v1.csv",
        "5C4E3C875B65BFD892FEABA6E3DF075862E5FA6E80862A78D3C7506D2FFDA51E",
        "53D601F8B3367AA69F7570BEA57BD2AB67E3FFB3631FAD1F3562EBB43309F1D2",
        "7CC7EBA73066AD5F66311FD284DEA4AB0975045B9B3A984335782ACD685149A8",
    ),
    (
        "F05b", "05_low_speed_tradeoff", "F05b_v2g_v4n_low_speed_tilt_tradeoff_v1",
        "05_low_speed_tradeoff/data/F05b_v2g_v4n_low_speed_tilt_tradeoff_data_v1.csv",
        "E0B622EC051DDE3DC107F2E83AAFDB8E6CC03FBEF49DCC51DF90EC6F80A7C421",
        "25960180F0603FAF888BA77B2DBF6034A8D5F071E14D5C2F20124412B89CD7FF",
        "B25D5D76B70BF60C41CFCAE3D627107923D248FF32B62BBC782F332F0B6474F7",
    ),
    (
        "F06a", "06_handoff_brake", "F06a_v2g_v4n_handoff_planar_speed_transient_v2",
        "06_handoff_brake/data/F06a_v2g_v4n_handoff_planar_speed_transient_data_v2.csv",
        "A4CF1BE7419B040201A0DCDB0E2438FB2C553FD9EB2E3F78BC6B94572F1420A7",
        "56753D0586727E48482954B3087925ACF529523DC763482A941365C29E63154B",
        "12873A50391623F11DF38326AFFEFFE3096CC267A61A0FE887BC6C08AF993196",
    ),
    (
        "F06b", "06_handoff_brake", "F06b_v2g_v4n_handoff_body_tilt_transient_v1",
        "06_handoff_brake/data/F06b_v2g_v4n_handoff_body_tilt_transient_data_v1.csv",
        "F9F392A8CBA2C865B2F52825DE93855D12A2EDC22101019E98E0AC0A2513C943",
        "0499225099E3DCA195B98ADEC0EA405801A6C0D2FEB7C9C5748565E698404FD5",
        "B19F34C3C4B241C770F23F8F00112855AEEE1563C990940BA84E397E4E77DCB1",
    ),
    (
        "F07a", "07_integrated_router", "F07a_integrated_router_sequence_v4",
        "07_integrated_router/data/F07a_integrated_router_sequence_data_v4.csv",
        "59297050223CA54DF6B9C87277E4847247D5444B1014C329F2F981B1253DF074",
        "4FB7001BE74719F9C86E9C9616C81658322C555AACB9B0AF048BEA8B2D3EFCAD",
        "A0622B27844BC4146ABA9A8C7A26710602D04FA2348A7AC0B87BFE85354E7123",
    ),
]


# old_rel, sha, canonical_target_rel (or None for intentionally superseded)
OBSOLETE = [
    (
        "01_v2_v3_evolution/F01a_v2v3_straight_tracking_error_data_v3.csv",
        "5F8D83A41E28E54BD97EA82E6AE0B4ABFA0174AE4355EB8DB3FB87B8ACABFDCF",
        "01_v2_v3_evolution/F01a_v2v3_straight_tracking_error_data_v4.csv",
    ),
    (
        "01_v2_v3_evolution/F01b_v2v3_straight_mean_tilt_data_v1.csv",
        "43ABB668D9F1DC04F093695EF330BC9B8566546606E3678E0BA8DC744764D2D4",
        "01_v2_v3_evolution/F01b_v2v3_straight_mean_tilt_data_v2.csv",
    ),
    (
        "01_v2_v3_evolution/F01c_v2a_vs_v2f_yaw_tracking_data_v1.csv",
        "4BCDDBDCC25CCAB9A93C23F3FDCC17EC16ED071690CA5FFFEBAAE533E65AFACB",
        "01_v2_v3_evolution/F01c_v2a_vs_v2f_yaw_tracking_data_v2.csv",
    ),
    (
        "02_v4_campaign/data/F02a_v4_training_episode_length_data_v2.csv",
        "499E341FD52B3F5D3E4A8E772E07B4950F360DDC862A12A8A7075025C12F184E",
        "02_v4_campaign/data/F02a_v4_training_episode_length_data_v3.csv",
    ),
    (
        "02_v4_campaign/data/F02a_v4_training_episode_length_zoom_data_v1.csv",
        "499E341FD52B3F5D3E4A8E772E07B4950F360DDC862A12A8A7075025C12F184E",
        "02_v4_campaign/data/F02a_v4_training_episode_length_zoom_data_v2.csv",
    ),
    (
        "02_v4_campaign/data/F02b_v4b_v4n_cross_track_drift_data_v1.csv",
        "812A243F737B9E701AA2993EA24F649E84CA2351196309F5EACA4E85F95F04BF",
        "02_v4_campaign/data/F02b_v4b_v4n_cross_track_drift_data_v3.csv",
    ),
    (
        "02_v4_campaign/data/F02b_v4b_v4n_cross_track_drift_data_v2.csv",
        "812A243F737B9E701AA2993EA24F649E84CA2351196309F5EACA4E85F95F04BF",
        "02_v4_campaign/data/F02b_v4b_v4n_cross_track_drift_data_v3.csv",
    ),
    (
        "02_v4_campaign/data/F02c_v4b_v4n_knee_kinematic_asymmetry_data_v1.csv",
        "FD770EC9FFAE93E06E109095A591033DE0DBAA81FE86DEAEC7E7027C7A6BA1CE",
        "02_v4_campaign/data/F02c_v4b_v4n_knee_kinematic_asymmetry_data_v3.csv",
    ),
    (
        "02_v4_campaign/data/F02c_v4b_v4n_knee_kinematic_asymmetry_data_v2.csv",
        "FD770EC9FFAE93E06E109095A591033DE0DBAA81FE86DEAEC7E7027C7A6BA1CE",
        "02_v4_campaign/data/F02c_v4b_v4n_knee_kinematic_asymmetry_data_v3.csv",
    ),
    (
        "03_v4n_final_gait/data/F03a_v4n_gait_cycle_foot_clearance_contact_data_v1.csv",
        "1F3C1600BF7A673EA72E72412A6FAADD57FCF66CAE5730FD84AD03F7FE171818",
        "03_v4n_final_gait/data/F03a_v4n_gait_cycle_foot_clearance_contact_data_v3.csv",
    ),
    (
        "03_v4n_final_gait/data/F03a_v4n_gait_cycle_foot_clearance_contact_data_v2.csv",
        "1F3C1600BF7A673EA72E72412A6FAADD57FCF66CAE5730FD84AD03F7FE171818",
        "03_v4n_final_gait/data/F03a_v4n_gait_cycle_foot_clearance_contact_data_v3.csv",
    ),
    (
        "04_v2g_final_stand/data/F04a_v2g_standing_planar_trajectory_data_v2.csv",
        "B9AEF6F7EA854F6F5B846D32EC2C91E318101A4BAE1D65A9021094ABD708586F",
        "04_v2g_final_stand/data/F04a_v2g_standing_planar_trajectory_data_v3.csv",
    ),
    (
        "04_v2g_final_stand/data/F04b_v2g_standing_body_tilt_data_v1.csv",
        "3C12E2F209FF0F3A97EC24EF97DE51C2A34C6290156A56C725623C880611E550",
        "04_v2g_final_stand/data/F04b_v2g_standing_body_tilt_data_v2.csv",
    ),
    (
        "06_handoff_brake/data/F06a_v2g_v4n_handoff_planar_speed_transient_data_v1.csv",
        "12873A50391623F11DF38326AFFEFFE3096CC267A61A0FE887BC6C08AF993196",
        "06_handoff_brake/data/F06a_v2g_v4n_handoff_planar_speed_transient_data_v2.csv",
    ),
    (
        "07_integrated_router/data/F07a_integrated_router_sequence_data_v1.csv",
        "A0622B27844BC4146ABA9A8C7A26710602D04FA2348A7AC0B87BFE85354E7123",
        "07_integrated_router/data/F07a_integrated_router_sequence_data_v4.csv",
    ),
    (
        "07_integrated_router/data/F07a_integrated_router_sequence_data_v2.csv",
        "A0622B27844BC4146ABA9A8C7A26710602D04FA2348A7AC0B87BFE85354E7123",
        "07_integrated_router/data/F07a_integrated_router_sequence_data_v4.csv",
    ),
    (
        "07_integrated_router/data/F07a_integrated_router_sequence_data_v3.csv",
        "A0622B27844BC4146ABA9A8C7A26710602D04FA2348A7AC0B87BFE85354E7123",
        "07_integrated_router/data/F07a_integrated_router_sequence_data_v4.csv",
    ),
    (
        "04_v2g_final_stand/pdf/F04a_v2g_standing_planar_drift_v1.pdf",
        "B3C63A679F09E1DCF9969A271CB8381530C46395EEC7F102125F12EB0A2C94C0",
        None,
    ),
    (
        "04_v2g_final_stand/data/F04a_v2g_standing_planar_drift_data_v1.csv",
        "F600807BFA2E032E839B2BFAC35D64A8C11A4418232864C3B32BA9044F33DD46",
        None,
    ),
]

SOURCE_DATASETS_TO_KEEP = [
    "01_v2_v3_evolution/data/straight_common_eval_v1",
    "01_v2_v3_evolution/data/yaw_common_eval_v1",
]


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def canonical_paths(root: Path):
    for fig_id, folder, stem, data_rel, png_sha, pdf_sha, data_sha in CANONICAL:
        yield fig_id, "PNG", root / folder / "png" / f"{stem}.png", png_sha
        yield fig_id, "PDF", root / folder / "pdf" / f"{stem}.pdf", pdf_sha
        yield fig_id, "DATA", root / data_rel, data_sha


def verify_canonical(root: Path) -> list[str]:
    lines = []
    for fig_id, kind, path, expected_sha in canonical_paths(root):
        if not path.is_file():
            fail(f"{fig_id} {kind}: missing canonical file: {path}")
        actual = sha256_file(path)
        if actual != expected_sha:
            fail(
                f"{fig_id} {kind}: SHA mismatch\n"
                f"expected={expected_sha}\nactual  ={actual}\npath={path}"
            )
        lines.append(f"{fig_id} {kind} SHA PASS")
    return lines


def verify_obsolete_preflight(root: Path) -> tuple[list[tuple[str, str, str | None]], list[str]]:
    present = []
    lines = []
    for rel, expected_sha, canonical_rel in OBSOLETE:
        old = root / rel
        if not old.exists():
            lines.append(f"ALREADY ABSENT: {rel}")
            continue
        if not old.is_file():
            fail(f"obsolete path is not a file: {old}")

        actual = sha256_file(old)
        if actual != expected_sha:
            fail(
                f"obsolete file SHA mismatch; refusing cleanup:\n"
                f"path={old}\nexpected={expected_sha}\nactual={actual}"
            )

        if canonical_rel is not None:
            target = root / canonical_rel
            if not target.is_file():
                fail(f"canonical replacement missing for {rel}: {target}")
            target_sha = sha256_file(target)
            if target_sha != expected_sha:
                fail(
                    f"duplicate verification failed for {rel}:\n"
                    f"old SHA={actual}\ncanonical SHA={target_sha}"
                )
            lines.append(f"DUPLICATE VERIFIED: {rel} == {canonical_rel}")
        else:
            lines.append(f"SUPERSEDED SHA VERIFIED: {rel}")

        present.append((rel, expected_sha, canonical_rel))
    return present, lines


def strict_post_audit(root: Path) -> list[str]:
    lines = []

    # 1) All 51 canonical artifacts must still match their audited hashes.
    lines.extend(verify_canonical(root))

    # 2) No obsolete path may remain.
    leftovers = [rel for rel, _, _ in OBSOLETE if (root / rel).exists()]
    if leftovers:
        fail("obsolete files still present after cleanup:\n" + "\n".join(leftovers))
    lines.append("OBSOLETE_PATHS_ABSENT_PASS")

    # 3) Exactly 17 canonical PNG and exactly 17 canonical PDF files.
    expected_png = {
        (Path(folder) / "png" / f"{stem}.png").as_posix()
        for _, folder, stem, _, _, _, _ in CANONICAL
    }
    expected_pdf = {
        (Path(folder) / "pdf" / f"{stem}.pdf").as_posix()
        for _, folder, stem, _, _, _, _ in CANONICAL
    }

    actual_png = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*.png")
        if p.parent.name == "png"
    }
    actual_pdf = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*.pdf")
        if p.parent.name == "pdf"
    }

    if actual_png != expected_png:
        extra = sorted(actual_png - expected_png)
        missing = sorted(expected_png - actual_png)
        fail(f"PNG canonical-set mismatch\nextra={extra}\nmissing={missing}")
    if actual_pdf != expected_pdf:
        extra = sorted(actual_pdf - expected_pdf)
        missing = sorted(expected_pdf - actual_pdf)
        fail(f"PDF canonical-set mismatch\nextra={extra}\nmissing={missing}")

    lines.append(f"PNG_CANONICAL_SET_PASS count={len(actual_png)}")
    lines.append(f"PDF_CANONICAL_SET_PASS count={len(actual_pdf)}")

    # 4) Exactly 17 figure data-freezes matching *_data_vN.csv.
    expected_data = {Path(row[3]).as_posix() for row in CANONICAL}
    pattern = re.compile(r"_data_v\d+\.csv$", re.IGNORECASE)
    actual_data = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*.csv")
        if pattern.search(p.name)
    }
    if actual_data != expected_data:
        extra = sorted(actual_data - expected_data)
        missing = sorted(expected_data - actual_data)
        fail(f"DATA freeze canonical-set mismatch\nextra={extra}\nmissing={missing}")
    lines.append(f"DATA_FREEZE_CANONICAL_SET_PASS count={len(actual_data)}")

    # 5) Raw standardized source evaluation datasets must remain intact/present.
    for rel in SOURCE_DATASETS_TO_KEEP:
        p = root / rel
        if not p.is_dir():
            fail(f"source evaluation dataset missing: {p}")
    lines.append("F01_SOURCE_EVALUATION_DATASETS_PRESENT_PASS")

    return lines


def write_report(
    root: Path,
    archive_dir: Path | None,
    moved: list[tuple[str, str]],
    preflight_lines: list[str],
    post_lines: list[str],
) -> Path:
    report = root / "THESIS_FIGURES_FINAL_FREEZE_AUDIT.txt"
    script_sha = sha256_file(Path(__file__)) if Path(__file__).is_file() else "N/A"

    lines = [
        "THESIS FIGURES FINAL — FREEZE AUDIT",
        "=" * 104,
        "",
        "STATUS = PASS / PACKAGE FROZEN",
        "",
        f"Root               : {root}",
        f"Archive            : {archive_dir if archive_dir else 'NONE / already clean'}",
        f"Cleanup script SHA : {script_sha}",
        f"Canonical figures  : {len(CANONICAL)}",
        f"Moved obsolete     : {len(moved)}",
        "",
        "PREFLIGHT",
        "-" * 104,
        *preflight_lines,
        "",
        "MOVED FILES",
        "-" * 104,
    ]
    if moved:
        for rel, sha in moved:
            lines.append(f"{rel}  SHA256={sha}")
    else:
        lines.append("NONE — package was already clean.")

    lines += [
        "",
        "POST-CLEANUP STRICT AUDIT",
        "-" * 104,
        *post_lines,
        "",
        "FINAL COUNTS",
        "  canonical PNG        = 17",
        "  canonical PDF        = 17",
        "  canonical data freeze= 17",
        "  obsolete residual    = 0",
        "",
        "SCIENTIFIC DATA / METRICS / FIGURE CONTENT = UNCHANGED",
        "REPOSITORY SOURCE / CHECKPOINTS = NOT MODIFIED",
        "",
        PASS_MARKER,
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Safely archive obsolete thesis-figure artifacts and freeze the canonical "
            "17-figure package after exact SHA256 validation."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive parent. Default: <root parent>/figures_archive",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move verified obsolete files. Without this flag: preflight only.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.is_dir():
        fail(f"figures_final root missing: {root}")

    print("=" * 104)
    print("THESIS FIGURES FINAL — CLEANUP / FREEZE")
    print("=" * 104)
    print(f"Root   : {root}")
    print(f"Mode   : {'APPLY' if args.apply else 'PREFLIGHT ONLY'}")
    print()

    canonical_pre = verify_canonical(root)
    obsolete_present, obsolete_lines = verify_obsolete_preflight(root)
    preflight_lines = [
        "CANONICAL_SHA_PREFLIGHT_PASS",
        *obsolete_lines,
    ]

    print(f"Canonical artifacts verified : {len(canonical_pre)} / {len(CANONICAL) * 3}")
    print(f"Obsolete files present       : {len(obsolete_present)} / {len(OBSOLETE)}")

    if not args.apply:
        print()
        print("No files moved. Re-run with --apply after reviewing this preflight.")
        print("G1_THESIS_FIGURES_FINAL_CLEANUP_PREFLIGHT_PASS")
        return

    moved: list[tuple[str, str]] = []
    archive_dir: Path | None = None

    if obsolete_present:
        archive_root = args.archive_root or (root.parent / "figures_archive")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = archive_root / f"figures_final_pre_freeze_cleanup_{stamp}"
        if archive_dir.exists():
            fail(f"archive destination already exists: {archive_dir}")

        archive_dir.mkdir(parents=True, exist_ok=False)

        try:
            for rel, expected_sha, _ in obsolete_present:
                src = root / rel
                dst = archive_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

                if not dst.is_file():
                    fail(f"archive move failed: {dst}")
                if sha256_file(dst) != expected_sha:
                    fail(f"archived-file SHA mismatch after move: {dst}")
                moved.append((rel, expected_sha))

            archive_manifest = archive_dir / "MOVED_FILES_MANIFEST.txt"
            archive_manifest.write_text(
                "\n".join(
                    [
                        "THESIS FIGURES FINAL — PRE-FREEZE ARCHIVE",
                        "=" * 104,
                        "",
                        *[f"{rel}  SHA256={sha}" for rel, sha in moved],
                        "",
                        f"MOVED_FILE_COUNT={len(moved)}",
                        "ARCHIVE_MANIFEST_PASS",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            # Best-effort rollback of files moved in this run.
            for rel, _ in reversed(moved):
                archived = archive_dir / rel
                original = root / rel
                if archived.is_file() and not original.exists():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(archived), str(original))
            raise

    post_lines = strict_post_audit(root)
    report = write_report(root, archive_dir, moved, preflight_lines, post_lines)

    print()
    print("POST-CLEANUP AUDIT")
    print(f"Canonical PNG             : 17 / 17")
    print(f"Canonical PDF             : 17 / 17")
    print(f"Canonical data-freeze     : 17 / 17")
    print(f"Obsolete residual         : 0 / {len(OBSOLETE)}")
    print(f"Moved to reversible archive: {len(moved)}")
    if archive_dir:
        print(f"Archive                   : {archive_dir}")
    print(f"Freeze audit              : {report}")
    print("Scientific data changed   : NO")
    print("Repository/checkpoints    : NOT MODIFIED")
    print()
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
