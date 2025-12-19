use std::fs;
use std::path::Path;
use zvd_lib::codec::{Decoder, Frame, ProResDecoder};
use zvd_lib::format::Packet;
use zvd_lib::util::Buffer;

#[test]
fn decode_first_frame_from_prores_sample() {
    let path = Path::new("tests/files/HELLDIVERS2_prores_frame.bin");
    if !path.exists() {
        eprintln!(
            "Skipping ProRes decode smoke test; sample not present at {}",
            path.display()
        );
        return;
    }

    let data = fs::read(path).expect("failed to read ProRes sample frame");
    let mut packet = Packet::new(0, Buffer::from_vec(data));
    packet.set_keyframe(true);
    let mut decoder = ProResDecoder::new();

    decoder
        .send_packet(&packet)
        .expect("send_packet should succeed");
    match decoder.receive_frame() {
        Ok(Frame::Video(vf)) => {
            assert!(vf.width > 0 && vf.height > 0);
            assert!(!vf.data.is_empty());
        }
        other => panic!("Failed to decode a ProRes frame from sample: {:?}", other),
    }
}
