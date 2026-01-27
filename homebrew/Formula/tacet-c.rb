class TacetC < Formula
  desc "C/C++ bindings for tacet timing oracle library"
  homepage "https://github.com/agucova/tacet"
  version "0.3.2"
  license "MPL-2.0"

  if Hardware::CPU.arm?
    url "https://github.com/agucova/tacet/releases/download/v#{version}/libtacet_c-darwin-arm64.a"
    sha256 "84d9c7dfdd487b94d8d44084fe0a32436b6212cb903684a39686d92dff6525f5"
  else
    url "https://github.com/agucova/tacet/releases/download/v#{version}/libtacet_c-darwin-amd64.a"
    sha256 "f8143284bdd3013e33f555fb0fd91a0d4ebf0c5c26390c42f45672d9900f7b54"
  end

  # Resources for additional files needed
  resource "tacet.h" do
    url "https://github.com/agucova/tacet/releases/download/v#{version}/tacet.h"
    sha256 "d748c1b6676ba54f9e14226ab9f1606fa64cb5bcca032d99b9edfc42701cce46"
  end

  resource "tacet.hpp" do
    url "https://github.com/agucova/tacet/releases/download/v#{version}/tacet.hpp"
    sha256 "ec210a8025045bec859bda4a0ff96458fd91a6c39f026184401af3f0576d794e"
  end

  def install
    # The main URL downloads the static library
    if Hardware::CPU.arm?
      lib.install "libtacet_c-darwin-arm64.a" => "libtacet_c.a"
    else
      lib.install "libtacet_c-darwin-amd64.a" => "libtacet_c.a"
    end

    # Download and install headers
    resource("tacet.h").stage do
      (include/"tacet").install "tacet.h"
    end

    resource("tacet.hpp").stage do
      (include/"tacet").install "tacet.hpp"
    end

    # Generate pkg-config file
    (lib/"pkgconfig").mkpath
    (lib/"pkgconfig/tacet.pc").write <<~EOS
      prefix=#{prefix}
      exec_prefix=${prefix}
      libdir=#{lib}
      includedir=#{include}/tacet

      Name: tacet
      Description: Statistical timing side-channel detection library
      Version: #{version}
      URL: https://github.com/agucova/tacet
      Libs: -L${libdir} -ltacet_c -framework Security -framework CoreFoundation
      Cflags: -I${includedir}
    EOS
  end

  test do
    # Test pkg-config
    assert_match version.to_s, shell_output("pkg-config --modversion tacet")
    assert_match "-ltacet_c", shell_output("pkg-config --libs tacet")
    assert_match "-I#{include}/tacet", shell_output("pkg-config --cflags tacet")

    # Test C compilation
    (testpath/"test.c").write <<~EOS
      #include <tacet/tacet.h>
      #include <stdio.h>

      int main() {
          const char* version = to_version();
          printf("tacet version: %s\\n", version);

          ToConfig cfg = to_config_adjacent_network();
          to_config_free(cfg);

          return 0;
      }
    EOS

    system ENV.cc, "test.c", "-o", "test",
           "-I#{include}/tacet", "-L#{lib}", "-ltacet_c",
           "-framework", "Security", "-framework", "CoreFoundation"

    system "./test"

    # Test C++ compilation
    (testpath/"test.cpp").write <<~EOS
      #include <tacet/tacet.hpp>
      #include <iostream>

      int main() {
          std::cout << "tacet C++ wrapper test" << std::endl;

          auto oracle = tacet::Oracle()
              .attacker_model(tacet::AttackerModel::AdjacentNetwork)
              .build();

          return 0;
      }
    EOS

    system ENV.cxx, "-std=c++20", "test.cpp", "-o", "test_cpp",
           "-I#{include}/tacet", "-L#{lib}", "-ltacet_c",
           "-framework", "Security", "-framework", "CoreFoundation"

    system "./test_cpp"
  end
end
