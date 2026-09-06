"""Captured OG bootstrap boundary and sequence-window regression checks."""
import json
from pathlib import Path
import struct
import sys
import types
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'shared'))
from wulfram import handlers
from wulfram.server_net import NetMixin

class BootstrapBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        capture=json.loads((root/'rebuild/analysis/udp-bootstrap/capture.json').read_text())
        cls.identify=bytes.fromhex(capture['packets'][0]['hex'])
        cls.handshake=bytes.fromhex(capture['packets'][1]['hex'])
    def test_captured_batch_and_ack(self):
        parser=NetMixin();parser.debug_viewpoint=False
        ack=bytes.fromhex('020055667788')
        self.assertEqual(list(parser._parse_udp_datagram(self.identify+self.handshake+ack)),[self.identify,self.handshake,ack])
        parsed=handlers._parse_empirical_client_d_handshake(self.handshake)
        self.assertEqual((parsed['receive_window'],parsed['consumed'],len(parsed['streams'])),(15,200,2))
    def test_truncation_and_invalid_fields(self):
        for n in range(len(self.handshake)):
            with self.subTest(length=n):self.assertIsNone(handlers._parse_empirical_client_d_handshake(self.handshake[:n]))
        for value in [0,32768,0xffffffff]:
            bad=self.handshake[:5]+struct.pack('>I',value)+self.handshake[9:]
            self.assertIsNone(handlers._parse_empirical_client_d_handshake(bad))
        self.assertIsNone(handlers._parse_empirical_client_d_handshake(self.handshake+b'\0'))
        self.assertEqual(handlers._parse_empirical_client_d_handshake(self.handshake+b'\0',allow_trailing=True)['consumed'],200)
    def test_builder_window_independent_of_player(self):
        for player in [0,1342,65535,0xffffffff]:
            ctx=types.SimpleNamespace(session=types.SimpleNamespace(player_id=player),client_id=3)
            parsed=handlers._parse_empirical_client_d_handshake(handlers._build_server_d_handshake(ctx))
            self.assertEqual(parsed['receive_window'],15)
    def test_identify_string_excludes_length(self):
        parser=NetMixin();ctx=types.SimpleNamespace(client_id=7,session=types.SimpleNamespace());bound=[]
        parser.udp_addr_to_client={};parser.session_key_to_client={'Hello There':ctx}
        parser._bind_udp_client=lambda client,addr,**kw:bound.append((client,addr))
        parser._handle_single_udp_packet(None,self.identify,('127.0.0.1',1234))
        self.assertEqual(bound,[(ctx,('127.0.0.1',1234))])
if __name__=='__main__':unittest.main()
