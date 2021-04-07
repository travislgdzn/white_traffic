from scapy.all import *
import os
import json


class Sniffer:
    def __init__(self, filedFile = "fields.json"):
        self.fields_ = None
        self.packets_ = None

        with open(filedFile, "r") as f:
            self.fields_ = json.load(f)

    def Sniff(self, count = 200, iface = conf.iface):
        self.packets_ = sniff(iface = iface, count = count)
        return self.packets_

    def Rdpcap(self, pcapFile):
        self.packets_ = rdpcap(pcapName)
        return self.packets_

    def W2pcap(self, name = "demo.pcap"):
        if self.packets_ is None:
            raise Exception("No packets!")

        wrpcap(name, self.packets_)

    def pcap2csv(self, pcapName = "demo.pcap", csvName = "demo.csv"):
        command = "tshark -r {} -T fields".format(pcapName)
        for field in self.fields_:
            args = ["{}.{}".format(field, value) for value in self.fields_[field]]
            command += " " + " ".join(["-e " + arg for arg in args])

        command += " -E header=y -E separator=, -E quote=d -E occurrence=f > {}".format(csvName)
        os.system(command)

    # def CheckField(self):
    #     if self.packets_ is None:
    #         raise Exception("No packets!")

    #     all_layers = []
    #     for packet in self.packets_:
    #         for layer in packet.layers():
    #             if layer not in all_layers:
    #                 all_layers.append(layer)
        
    #     all_fields = list(set([layer.__name__.lower() for layer in all_layers]))
    #     resolved = [layer_ for layer_ in all_fields if layer_ in self.fields_]
    #     unresolved = [layer_ for layer_ in all_fields if layer_ not in self.fields_]

    #     print("unresolved:", unresolved)

if __name__ == '__main__':
    sniffer = Sniffer()

    # convert test.pcap to test.csv
    sniffer.pcap2csv("test.pcap", "test.csv")




'''
tshark -r demo.pcap -T fields -e eth.src -e eth.dst -e ip.src -e ip.dst -e ip.proto -E header=y -E separator=, -E quote=d -E occurrence=f > test.csv
'''