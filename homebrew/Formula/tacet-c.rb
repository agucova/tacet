class TacetC < Formula
  desc "C/C++ bindings for tacet timing oracle library"
  homepage "https://github.com/agucova/tacet"
  version "0.3.0"
  license "MPL-2.0"

  if Hardware::CPU.arm?
    url "https://github.com/agucova/tacet/releases/download/v#{version}/libtacet_c-darwin-arm64.a"
    sha256 "SHA256_ARM64_PLACEHOLDER" # Update with: shasum -a 256 libtacet_c-darwin-arm64.a
  else
    url "https://github.com/agucova/tacet/releases/download/v#{version}/libtacet_c-darwin-amd64.a"
    sha256 "SHA256_AMD64_PLACEHOLDER" # Update with: shasum -a 256 libtacet_c-darwin-amd64.a
  end

  # Resources for additional files needed
  resource "tacet.h" do
    url "https://github.com/agucova/tacet/releases/download/v#{version}/tacet.h"
    sha256 "SHA256_HEADER_PLACEHOLDER" # Update with: shasum -a 256 tacet.h
  end

  resource "tacet.hpp" do
    url "https://github.com/agucova/tacet/releases/download/v#{version}/tacet.hpp"
    sha256 "SHA256_CPP_HEADER_PLACEHOLDER" # Update with: shasum -a 256 tacet.hpp
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
